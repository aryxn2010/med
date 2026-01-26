import os
import time
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from werkzeug.utils import secure_filename
from datetime import datetime
import json
import pathlib
import boto3
import numpy as np # Ensure numpy is imported
import os
from PIL import Image, ImageDraw, ImageFont
from flask import send_from_directory
import re
# --- CONFIGURATION ---
import requests
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from google.cloud import firestore # NEW IMPORT
import uuid
import traceback
from google.oauth2 import service_account
# 1. Force the file path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Load Environment Variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Load Gemini API Key
api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
    genai.configure(api_key=api_key)
else:
    print("❌ CRITICAL ERROR: API Key not found! Set GEMINI_API_KEY in .env")

# Initialize Firestore (Robust Explicit Auth)
import base64 # Add this to imports if missing

# Initialize Firestore (Robust Explicit Auth)
print("\n--- GOOGLE CLOUD AUTH DIAGNOSTICS ---")

# 1. Try loading from Railway Environment Variable (SECURE METHOD)
encoded_key = os.environ.get("FIREBASE_BASE64_KEY")
key_path = os.path.join(BASE_DIR, "service-account.json")

try:
    if encoded_key:
        print("✅ Found FIREBASE_BASE64_KEY in Environment Variables")
        # Decode the base64 string back to JSON
        decoded_json = base64.b64decode(encoded_key).decode("utf-8")
        creds_dict = json.loads(decoded_json)
        
        # Load credentials directly from the dictionary
        creds = service_account.Credentials.from_service_account_info(creds_dict)
        db = firestore.Client(credentials=creds, project=creds.project_id)
        print(f"✅ Firestore Connected via Env Var: {db.project}")

    # 2. Fallback to local file (FOR LOCAL TESTING ONLY)
    elif os.path.exists(key_path):
        print(f"⚠️ Using local key file at: {key_path}")
        creds = service_account.Credentials.from_service_account_file(key_path)
        db = firestore.Client(credentials=creds, project=creds.project_id)
        print(f"✅ Firestore Connected via File: {db.project}")

    else:
        print("❌ ERROR: Authentication not found (No Env Var or Local File).")
        db = None

except Exception as e:
    print(f"\n❌ CRITICAL AUTH ERROR: {e}")
    traceback.print_exc()
    db = None
    
    
print("-------------------------------------\n")    
MODEL_NAME = "gemini-3-pro-preview"

def clean_json_response(text):
    """
    Clean and fix malformed JSON responses from Gemini.
    Handles unterminated strings, unescaped quotes, and other common issues.
    """
    if not text:
        return None
    
    # Remove markdown code blocks if present
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()
    
    # Try to find JSON object boundaries
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        text = text[start_idx:end_idx + 1]
    elif start_idx != -1:
        # If we have start but no end, try to find a reasonable end
        # Look for the last closing brace or add one
        text = text[start_idx:]
        if not text.endswith('}'):
            # Try to balance braces
            open_count = text.count('{')
            close_count = text.count('}')
            if open_count > close_count:
                text += '}' * (open_count - close_count)
    
    # Fix unterminated strings - more robust approach
    # Process character by character to properly handle string boundaries
    result = []
    in_string = False
    escape_next = False
    i = 0
    
    while i < len(text):
        char = text[i]
        
        if escape_next:
            result.append(char)
            escape_next = False
            i += 1
            continue
        
        if char == '\\':
            result.append(char)
            escape_next = True
            i += 1
            continue
        
        if char == '"':
            result.append(char)
            in_string = not in_string
            i += 1
            continue
        
        result.append(char)
        i += 1
    
    # If we ended in a string, close it
    if in_string:
        result.append('"')
    
    text = ''.join(result)
    
    # Try to parse - if it fails, try more aggressive cleaning
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # More aggressive cleaning: remove trailing incomplete parts
        # Find the error position and truncate there
        error_pos = getattr(e, 'pos', None)
        if error_pos and error_pos < len(text):
            # Try truncating at the error position and finding the last complete JSON object
            truncated = text[:error_pos]
            # Find the last complete key-value pair or object
            last_comma = truncated.rfind(',')
            last_colon = truncated.rfind(':')
            
            if last_comma > last_colon and last_comma > 0:
                # Remove the incomplete part after last comma
                truncated = truncated[:last_comma]
                # Try to close the JSON properly
                if truncated.count('{') > truncated.count('}'):
                    truncated += '}'
                try:
                    return json.loads(truncated + '}')
                except:
                    pass
        
        # Last resort: try to extract JSON using regex
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if json_match:
            try:
                cleaned = json_match.group(0)
                # Ensure it's properly closed
                if cleaned.count('{') > cleaned.count('}'):
                    cleaned += '}' * (cleaned.count('{') - cleaned.count('}'))
                return json.loads(cleaned)
            except:
                pass
        
        return None

def safe_get_response_text(response):
    """
    Safely extract text from Gemini response, handling safety filters and errors.
    """
    try:
        # Check if response has candidates
        if not hasattr(response, 'candidates') or not response.candidates:
            raise ValueError("No candidates in response")
        
        candidate = response.candidates[0]
        
        # Check finish reason FIRST before trying to access text
        # This prevents the ValueError from accessing response.text when finish_reason is 2
        if hasattr(candidate, 'finish_reason'):
            finish_reason = candidate.finish_reason
            # Check if finish_reason is an enum or integer
            finish_reason_value = finish_reason.value if hasattr(finish_reason, 'value') else int(finish_reason) if isinstance(finish_reason, (int, str)) else finish_reason
            
            if finish_reason_value == 2:  # SAFETY
                # Log the actual safety ratings for debugging
                safety_info = []
                if hasattr(candidate, 'safety_ratings'):
                    for rating in candidate.safety_ratings:
                        category = getattr(rating, 'category', 'Unknown')
                        probability = getattr(rating, 'probability', 'Unknown')
                        threshold = getattr(rating, 'threshold', 'Unknown')
                        safety_info.append(f"{category}: {probability} (threshold: {threshold})")
                    print(f"🚫 [SAFETY] Safety ratings: {', '.join(safety_info) if safety_info else 'No ratings available'}")
                else:
                    print(f"🚫 [SAFETY] Finish reason: {finish_reason_value} (SAFETY) - No detailed ratings available")
                raise ValueError("Response was blocked by safety filters. Please try with different content.")
            elif finish_reason_value == 3:  # RECITATION
                raise ValueError("Response was blocked due to recitation policy.")
            elif finish_reason_value != 1:  # 1 = STOP (normal)
                print(f"⚠️ [SAFETY] Unexpected finish reason: {finish_reason_value}")
                raise ValueError(f"Response finished with reason: {finish_reason_value}")
        
        # Now safely try to get text (finish_reason check passed)
        # Try direct text access first (fastest)
        try:
            if hasattr(response, 'text'):
                text = response.text
                if text:
                    return text
        except ValueError:
            # If direct access fails (e.g., finish_reason 2), try alternative methods
            pass
        
        # Alternative: try to get from candidate content
        if hasattr(candidate, 'content') and candidate.content:
            if hasattr(candidate.content, 'parts'):
                text_parts = []
                for part in candidate.content.parts:
                    if hasattr(part, 'text') and part.text:
                        text_parts.append(part.text)
                if text_parts:
                    return ''.join(text_parts)
        
        raise ValueError("Could not extract text from response - no valid parts found")
        
    except ValueError:
        # Re-raise ValueError as-is (these are our custom errors)
        raise
    except Exception as e:
        # Wrap other exceptions
        raise ValueError(f"Error extracting response text: {str(e)}") 
import urllib.request
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
import uuid # <--- ADD THIS AT THE TOP WITH OTHER IMPORTS
# --- UPDATED SYSTEM INSTRUCTION (UNIVERSAL PATHOLOGICAL MATCHER) ---
# --- UPDATED SYSTEM INSTRUCTION (SIMPLE & PATIENT-FRIENDLY) ---
SYSTEM_INSTRUCTION = """You are a clinical audio analysis expert for respiratory medicine. Your task is to analyze audio recordings of breathing sounds, coughs, and respiratory patterns to identify potential respiratory conditions.

CLINICAL ANALYSIS PROTOCOL:
1. Listen carefully to the audio for characteristic sounds: wheezing (asthma), stridor (croup), crackles (pneumonia), dry/hacking coughs, wet/productive coughs, shortness of breath patterns.
2. Match audio patterns to known clinical presentations: Croup (barking seal-like cough), Asthma (wheezing, high-pitched sounds), Pneumonia (crackles, wet cough), Bronchitis (productive cough), Upper Respiratory Infection (nasal congestion sounds).
3. Calculate internal confidence score (0-100) based on pattern matching accuracy.
4. Provide clear, accurate medical assessment in simple language for patients.

OUTPUT FORMAT (Strict JSON):
{
  "valid_audio": true,
  "universal_match": {"disease_name": "Specific Clinical Condition Name", "similarity_score": 85},
  "severity": "Low/Moderate/High",
  "infection_type": "Viral/Bacterial/Chronic/Irritation/Allergic",
  "simple_explanation": "Clear explanation of what the condition is and what was detected in the audio, in plain language without medical jargon.",
  "audio_characteristics": "Detailed description of specific sounds heard: type of cough, breathing patterns, any distinctive sounds (wheezing, stridor, crackles, etc.).",
  "recommendation": "Specific, actionable medical advice based on the findings."
}

CRITICAL: Be accurate and specific. Identify the actual respiratory condition based on audio patterns."""

MEDICINE_SYSTEM_INSTRUCTION = f"""
You are an AI Clinical Expert. 
Analyze the image of medicines OR the user's text query.
TODAY'S DATE: {datetime.now().strftime('%Y-%m-%d')}

CRITICAL RULE: 
- If an image is provided, analyze the visual details (label, pill shape).
- If NO image is provided, answer ONLY based on the user's text prompt. Do NOT hallucinate or describe a non-existent image.

JSON OUTPUT FORMAT:
{{
  "medicines_found": [],
  "interaction_warning": "...",
  "safety_alert": "...",
  "answer_to_user": "Direct response to the user's health query.",
  "recommendation": "Clinical advice and next steps."
}}
"""
NUTRITION_SYSTEM_INSTRUCTION = """
You are a Clinical Dietitian. 
Analyze the food image OR the user's description.

CRITICAL RULE:
- Do NOT provide specific numbers (Calories, grams, etc.). 
- ONLY provide qualitative tags (e.g., 'High Protein', 'Low Sodium', 'High Sugar', 'Balanced').
- If NO image is provided, analyze based on the text description only.

JSON OUTPUT FORMAT:
{
  "food_items": ["List of distinct food items"],
  "nutritional_info": "Qualitative Tags Only (e.g., 'High Protein | Low Sodium | Gluten-Free')",
  "deficiency_alert": "Does this meal fail the user's specific health goal?",
  "answer_to_user": "Direct feedback on this specific meal relative to their goal.",
  "improvements": "Specific, actionable suggestions to make this meal healthier (e.g., 'Add a side of spinach for iron', 'Replace soda with water')."
}
"""
# --- REPLACES THE OLD WOUND INSTRUCTION ---
WOUND_SYSTEM_INSTRUCTION = """
You are an Advanced Clinical Diagnostic AI. 
Your task is to analyze clinical images including Skin Diseases, Infections, Wounds, and Allergic Reactions.

### DIAGNOSTIC PROTOCOL:
1. **Analyze the Visuals:**
   - **Dermatology:** Eczema, Psoriasis, Acne, Hives.
   - **Wounds/Infection:** Look for Redness (Erythema), Swelling (Edema), Pus (Purulence), Black tissue (Necrosis), or Healing tissue (Granulation).

2. **TIMELINE ANALYSIS (If Multiple Images):**
   - Treat Image 1 as "Day 1" (Baseline) and the last Image as "Current Status".
   - Compare them strictly. 
   - **Healing:** Redness reduced? Size smaller? Scab forming?
   - **Worsening:** Redness spreading? Pus increased? Black tissue appearing?

3. **Assess Severity:**
   - **Critical:** Sepsis signs (streaks), Necrosis, Deep open wounds. -> "Seek ER".
   - **Moderate:** Infection signs (pus/heat), deep cuts. -> "Visit Clinic".
   - **Low:** Healing scabs, minor cuts. -> "Home Care".

### JSON OUTPUT FORMAT:
{
  "condition_name": "Specific Diagnosis (e.g., 'Infected Abrasion', 'Diabetic Ulcer')",
  "category": "Medical Category",
  "severity": "Critical (ER) / Moderate (Clinic) / Low (Home Care)",
  "clinical_status": "Status (e.g., 'Active Infection', 'Granulating (Healing)', 'Deteriorating')",
  "healing_trend": "Improving / Worsening / Stagnant / N/A (Single Image)",
  "timeline_analysis": "Specific comparison from Day 1 to current. Example: 'Redness has decreased significantly compared to Day 1, indicating positive healing.'",
  "immediate_care": "Immediate steps (e.g., 'Clean with saline', 'Apply antibiotic ointment').",
  "answer_to_user": "Compassionate explanation of the symptoms.",
  "recommendation": "Actionable advice + Home Routine."
}
"""

def analyze_vision_with_gemini(image_paths, scan_type, user_prompt="", language='en'):    
    # Select System Prompt based on scan type
    if scan_type == "wound":
        sys_instruction = WOUND_SYSTEM_INSTRUCTION
    elif scan_type == "nutrition":
        sys_instruction = NUTRITION_SYSTEM_INSTRUCTION
    else:
        sys_instruction = MEDICINE_SYSTEM_INSTRUCTION 

    # Language Mapping
    lang_map = {"en": "English", "hi": "Hindi", "kn": "Kannada"}
    target_lang = lang_map.get(language, "English")

    content_payload = []
    uploaded_refs = []
    
    if image_paths:
        for index, path in enumerate(image_paths): # Changed to use enumerate
            img_file = genai.upload_file(path=path)
            while img_file.state.name == "PROCESSING":
                time.sleep(1)
                img_file = genai.get_file(img_file.name)
            
            # Add context for wound timeline
            if scan_type == "wound" and len(image_paths) > 1:
                content_payload.append(f"Image {index + 1} (Day {index + 1})")
            
            content_payload.append(img_file)
            uploaded_refs.append(img_file)
    
    content_payload.append(f"User Context: {user_prompt}")
    
    # Inject Language Instruction
    content_payload.append(f"CRITICAL: All text values in the JSON output MUST be in {target_lang}. Translate medical terms accurately into {target_lang}.")

    if not image_paths:
        content_payload.append(f"SYSTEM NOTICE: No image provided. Answer strictly based on text in {target_lang}.")

    safety_config_vision = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE  # <--- CHANGED
    }
    
    # Try models in order: pro first for accuracy, then flash
    models_to_try = [MODEL_NAME, "gemini-3-flash-preview"]
    response = None
    response_text = None
    
    for model_idx, model_name in enumerate(models_to_try):
        try:
            print(f"🤖 [VISION] Using {model_name} for vision analysis (attempt {model_idx + 1}/{len(models_to_try)})...")
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=sys_instruction,
                safety_settings=safety_config_vision
            )
            
            response = model.generate_content(
                content_payload,
                generation_config={
                    "response_mime_type": "application/json",
                    "max_output_tokens": 1500
                },
                safety_settings=safety_config_vision,
                request_options={"timeout": 120}
            )
            
            # Verify response is valid
            try:
                response_text = safe_get_response_text(response)
                break  # Success - exit loop
            except ValueError as ve:
                error_msg = str(ve).lower()
                if ("safety" in error_msg or "blocked" in error_msg) and model_idx < len(models_to_try) - 1:
                    print(f"🚫 [VISION] {model_name} blocked, trying next model...")
                    continue  # Try next model
                else:
                    # Clean up and re-raise
                    for ref in uploaded_refs:
                        try:
                            genai.delete_file(ref.name)
                        except:
                            pass
                    raise ValueError(f"Vision analysis response error: {ve}")
                    
        except ValueError as ve:
            error_msg = str(ve).lower()
            if ("safety" in error_msg or "blocked" in error_msg) and model_idx < len(models_to_try) - 1:
                print(f"🚫 [VISION] {model_name} blocked, trying next model...")
                continue
            else:
                # Clean up uploaded files on error
                for ref in uploaded_refs:
                    try:
                        genai.delete_file(ref.name)
                    except:
                        pass
                raise
        except Exception as e:
            if model_idx < len(models_to_try) - 1:
                print(f"⚠️ [VISION] Error with {model_name}, trying next model: {e}")
                continue
            # Clean up uploaded files on error
            for ref in uploaded_refs:
                try:
                    genai.delete_file(ref.name)
                except:
                    pass
            raise  # Re-raise to be handled by caller
    
    if not response_text:
        # Clean up and raise error
        for ref in uploaded_refs:
            try:
                genai.delete_file(ref.name)
            except:
                pass
        raise ValueError("All models blocked or failed for vision analysis")
    
    for ref in uploaded_refs:
        try:
            genai.delete_file(ref.name)
        except:
            pass

    # Safely extract and clean response
    try:
        response_text = safe_get_response_text(response)
        # Try to parse as JSON and re-stringify to ensure it's valid
        try:
            data = json.loads(response_text)
            return json.dumps(data)  # Return clean JSON
        except json.JSONDecodeError:
            # If not JSON, try to clean it
            cleaned = clean_json_response(response_text)
            if cleaned:
                return json.dumps(cleaned)
            # Fallback: return cleaned text
            return response_text.strip()
    except ValueError as e:
        raise ValueError(f"Vision analysis failed: {e}")


@app.route('/save_patient_data', methods=['POST'])
def save_patient_data():
    # 1. Immediate Safety Check
    if not db:
        return jsonify({"error": "Database not connected (Auth Failed)"}), 500

    new_record = request.json
    if 'id' not in new_record:
        new_record['id'] = str(uuid.uuid4())
    
    new_record['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    try:
        # 2. Add a Timeout to the write operation
        # This prevents the server from hanging forever if Auth is bad
        doc_ref = db.collection('patients').document(new_record['id'])
        
        # We perform the 'set' operation with a timeout constraint
        # Note: The set() method itself doesn't take a timeout, but we wrap the logic
        # by ensuring the db client is valid. If auth fails, this line throws the exception.
        doc_ref.set(new_record)
        
        return jsonify({"status": "success", "message": "Saved to Cloud", "id": new_record['id']})

    except Exception as e:
        print(f"❌ FIRESTORE WRITE ERROR: {e}")
        # Return a clear error to the frontend instead of crashing the server
        return jsonify({"error": "Failed to save data to cloud."}), 500
    
@app.route('/delete_patient_record', methods=['POST'])
def delete_patient_record():
    if not db:
        return jsonify({"error": "Database not connected"}), 500

    record_id = request.json.get('id')
    if not record_id:
        return jsonify({"error": "No ID provided"}), 400
    
    try:
        # Delete from Google Cloud Firestore
        db.collection('patients').document(record_id).delete()
        return jsonify({"status": "success", "deleted": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
def get_dynamic_font(size):
    """
    Loads the local Roboto-Regular.ttf file.
    """
    font_filename = "Roboto-Regular.ttf"
    
    # Check if the file exists in the current folder
    if os.path.exists(font_filename):
        try:
            return ImageFont.truetype(font_filename, size)
        except Exception as e:
            print(f"Error loading font: {e}")
            return ImageFont.load_default()
    else:
        # Fallback if you forgot to paste the file
        print("Roboto-Regular.ttf not found. Using default font.")
        return ImageFont.load_default()
# --- EXTREME LOGIC FORM INSTRUCTION (TYPESETTER ENGINE) ---
FORM_SYSTEM_INSTRUCTION = """
You are an Elite Document Forensics AI.
Your goal is "Typesetter Precision". You must identify EXACTLY where text should be physically typed on the page.

### EXTREME LOGIC PROTOCOL:
1. **Analyze the "Write Zone" (`value_rect`):**
   - STRICTLY identify the empty underline or box space.
   - **CRITICAL:** Do NOT include the label text (e.g., "Name:") in this rect. Start the rect 5 pixels AFTER the label ends.
   - The `value_rect` [ymin, xmin, ymax, xmax] must represent the *baseline* area where the ink touches the paper.

2. **Font Forensics (`font_style`):**
   - Look at the printed text on the form.
   - "serif" -> If letters have feet (Times New Roman, Georgia).
   - "sans" -> If letters are clean (Arial, Helvetica, Roboto).
   - "mono" -> If it looks like code or a typewriter (Courier).

3. **Data Extraction:**
   - Convert the user's voice/text input into professional medical terminology suitable for the field.

### JSON OUTPUT FORMAT:
{
  "visual_fields": [
    {
      "key": "Patient Name",
      "value": "Aryan Gupta",
      "font_style": "serif", 
      "value_rect": [155, 250, 185, 600], 
      "label_rect": [155, 50, 185, 240]
    }
  ],
  "confirmation_message": "Document aligned and filled."
}
"""


def get_smart_color(image, rect):
    """
    Samples the darkest pixels from the Label area to match ink color.
    """
    try:
        w, h = image.size
        ymin, xmin, ymax, xmax = rect
        
        # Crop the label area
        box = (int(xmin * w / 1000), int(ymin * h / 1000), int(xmax * w / 1000), int(ymax * h / 1000))
        crop = image.crop(box)
        
        # Convert to numpy array to find dark pixels
        arr = np.array(crop)
        # Filter: Ignore white/transparent background pixels (assuming > 200 is background)
        mask = np.all(arr[:, :, :3] < 200, axis=2)
        
        if np.sum(mask) > 0:
            # Get average color of the dark pixels
            avg_color = np.mean(arr[mask], axis=0).astype(int)
            return tuple(avg_color[:3]) # Return (R, G, B)
    except Exception as e:
        print(f"Color sampling failed: {e}")
    
    return (20, 20, 30) # Default to Soft Black/Dark Slate if sampling fails

def get_best_fit_font(text, font_path, max_width, initial_size):
    """
    Recursively shrinks font size until text fits within max_width.
    Safe-guards against missing font files by falling back to default.
    """
    # Check if font file actually exists
    if not os.path.exists(font_path):
        return ImageFont.load_default()

    size = initial_size
    min_size = 10 
    
    while size > min_size:
        try:
            font = ImageFont.truetype(font_path, size)
        except Exception:
            return ImageFont.load_default()
            
        # Measure text width
        length = font.getlength(text) if hasattr(font, 'getlength') else font.getsize(text)[0]
        
        if length < max_width:
            return font
            
        size -= 1 
        
    # If we exit loop (too small), try returning min size, OR default if that fails
    try:
        return ImageFont.truetype(font_path, min_size)
    except:
        return ImageFont.load_default()

@app.route('/get_patient_history', methods=['GET'])
def get_patient_history():
    if not db:
        print("DEBUG: Database variable is None.")
        return jsonify([])

    try:
        print("DEBUG: Attempting to fetch patients...")
        # Fetch from Firestore
        docs = db.collection('patients').order_by('timestamp', direction=firestore.Query.DESCENDING).stream()
        records = [doc.to_dict() for doc in docs]
        print(f"DEBUG: Success! Found {len(records)} records.")
        return jsonify(records)
    except Exception as e:
        # PRINT THE FULL GOOGLE ERROR
        print(f"\n❌ FIRESORE PERMISSION ERROR: {e}")
        # This will print details like "User X is missing permission Y on resource Z"
        return jsonify([])    
def analyze_form_voice(audio_path, text_input, mode, doc_path=None, language='en'):
    files_to_send = []
    lang_map = {"en": "English", "hi": "Hindi", "kn": "Kannada"}
    target_lang = lang_map.get(language, "English")

    # --- STEP 1: TRANSCRIBE AUDIO (OPTIMIZED) ---
    print(f"🎤 [FORM] Starting form voice processing...")
    form_start = time.time()
    final_text_input = text_input or ""
    audio_file = None
    
    if audio_path:
        try:
            print(f"📤 [FORM] Uploading audio for transcription...")
            # Upload audio file with extended timeout
            audio_file = genai.upload_file(path=audio_path)
            upload_timeout = 60  # Extended timeout
            upload_start = time.time()
            
            while audio_file.state.name == "PROCESSING":
                elapsed = time.time() - upload_start
                if elapsed > upload_timeout:
                    raise TimeoutError(f"Audio upload timeout after {elapsed:.1f}s")
                time.sleep(0.5)  # Check every 0.5s
                audio_file = genai.get_file(audio_file.name)
            
            upload_elapsed = time.time() - upload_start
            print(f"✅ [FORM] Audio uploaded in {upload_elapsed:.1f}s")
            
            safety_config_form = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
            }
            
            # Try models for transcription: pro first, then flash
            transcribe_models = [MODEL_NAME, "gemini-3-flash-preview"]
            transcribed_text = ""
            
            for transcribe_idx, transcribe_model_name in enumerate(transcribe_models):
                try:
                    print(f"🎙️ [FORM] Transcribing with {transcribe_model_name} (attempt {transcribe_idx + 1}/{len(transcribe_models)})...")
                    transcribe_model = genai.GenerativeModel(
                        transcribe_model_name,
                        safety_settings=safety_config_form
                    )
                    
                    transcribe_res = transcribe_model.generate_content(
                        [audio_file, f"This is a medical audio recording. Transcribe the spoken words exactly into {target_lang}. Return ONLY the transcribed text, no additional commentary or analysis."],
                        generation_config={"max_output_tokens": 500},
                        safety_settings=safety_config_form,
                        request_options={"timeout": 60}
                    )
                    
                    # Safely extract text with error handling
                    try:
                        transcribed_text = safe_get_response_text(transcribe_res).strip()
                        if transcribed_text:
                            print(f"✅ [FORM] Transcription successful: '{transcribed_text[:50]}...'")
                            break  # Success - exit loop
                    except ValueError as e:
                        error_msg = str(e).lower()
                        if ("safety" in error_msg or "blocked" in error_msg) and transcribe_idx < len(transcribe_models) - 1:
                            print(f"🚫 [FORM] {transcribe_model_name} blocked, trying next model...")
                            continue  # Try next model
                        else:
                            print(f"⚠️ [FORM] Transcription failed: {e}")
                            transcribed_text = ""
                            break
                except Exception as e:
                    if transcribe_idx < len(transcribe_models) - 1:
                        print(f"⚠️ [FORM] Error with {transcribe_model_name}, trying next: {e}")
                        continue
                    else:
                        print(f"⚠️ [FORM] All transcription models failed: {e}")
                        transcribed_text = ""
            # Add transcribed text to final input if available
            if transcribed_text:
                final_text_input += f" {transcribed_text}"
                transcribe_elapsed = time.time() - upload_start
                print(f"✅ [FORM] Transcription complete in {transcribe_elapsed:.1f}s: '{transcribed_text[:50]}...'")
            else:
                print(f"⚠️ [FORM] No transcription available, using text input only")
            
            # Clean up immediately after transcription
            if audio_file:
                try:
                    genai.delete_file(audio_file.name)
                except:
                    pass
            
        except Exception as e:
            print(f"❌ [FORM] Transcription Error: {e}")
            # Clean up on error
            if audio_file:
                try:
                    genai.delete_file(audio_file.name)
                except:
                    pass

    # --- STEP 2: PREPARE IMAGE (OPTIMIZED) ---
    if doc_path:
        doc_file = genai.upload_file(path=doc_path)
        upload_timeout = 25
        upload_start = time.time()
        
        while doc_file.state.name == "PROCESSING":
            if time.time() - upload_start > upload_timeout:
                raise TimeoutError("Document upload timeout")
            time.sleep(0.3)  # Faster checking
            doc_file = genai.get_file(doc_file.name)
        
        files_to_send.append(doc_file)

    # --- STEP 3: LOGIC & ALIGNMENT (OPTIMIZED) ---
    # Optimized prompt - shorter and more direct
    EXTREME_PROMPT = f"""Document Typesetter. Map "{final_text_input}" to form fields in {target_lang}.

TASK:
1. Text Fields: Map spoken values. "value" in {target_lang}. Start xmin AFTER label ends (4 spaces gap).
2. Checkboxes: Return exact inner boundary if requested.

BOUNDING BOX: [ymin, xmin, ymax, xmax] = blank writing space only. ymax = baseline.

JSON:
{{
  "visual_fields": [{{"key": "Field", "value": "Data", "value_rect": [ymin, xmin, ymax, xmax]}}],
  "checkbox_fields": [{{"key": "Label", "value_rect": [ymin, xmin, ymax, xmax]}}]
}}"""

    # Configure safety settings for form analysis
    safety_config_form = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
    }
    
    # Try models in order: pro first for accuracy, then flash
    models_to_try = [MODEL_NAME, "gemini-3-flash-preview"]
    response = None
    response_text = None
    
    for model_idx, model_name in enumerate(models_to_try):
        try:
            print(f"🤖 [FORM] Using {model_name} for form analysis (attempt {model_idx + 1}/{len(models_to_try)})...")
            model = genai.GenerativeModel(
                model_name=model_name,
                safety_settings=safety_config_form
            )
            
            analysis_start = time.time()
            response = model.generate_content(
                files_to_send + [EXTREME_PROMPT], 
                generation_config={
                    "response_mime_type": "application/json",
                    "max_output_tokens": 1200,
                    "temperature": 0.3
                },
                safety_settings=safety_config_form,
                request_options={"timeout": 120}
            )
            analysis_elapsed = time.time() - analysis_start
            print(f"✅ [FORM] Form analysis complete in {analysis_elapsed:.1f}s")
            
            # Safely extract and parse response
            try:
                response_text = safe_get_response_text(response)
                data = json.loads(response_text)
                break  # Success - exit loop
            except ValueError as ve:
                error_msg = str(ve).lower()
                if ("safety" in error_msg or "blocked" in error_msg) and model_idx < len(models_to_try) - 1:
                    print(f"🚫 [FORM] {model_name} blocked, trying next model...")
                    continue  # Try next model
                else:
                    raise
            except json.JSONDecodeError as e:
                print(f"⚠️ [FORM] JSON parse error, attempting to clean: {e}")
                data = clean_json_response(response_text)
                if data is None:
                    if model_idx < len(models_to_try) - 1:
                        continue  # Try next model
                    raise ValueError(f"Could not parse form analysis JSON: {e}")
                break  # Success
        except ValueError as ve:
            error_msg = str(ve).lower()
            if ("safety" in error_msg or "blocked" in error_msg) and model_idx < len(models_to_try) - 1:
                print(f"🚫 [FORM] {model_name} blocked, trying next model...")
                continue
            else:
                raise ValueError(f"Form analysis response error: {ve}")
        except Exception as e:
            if model_idx < len(models_to_try) - 1:
                print(f"⚠️ [FORM] Error with {model_name}, trying next model: {e}")
                continue
            raise
    
    if not response_text:
        raise ValueError("All models blocked or failed for form analysis")
        if "visual_fields" not in data: data["visual_fields"] = []
        if "checkbox_fields" not in data: data["checkbox_fields"] = []
    except Exception as e:
        print(f"❌ [FORM] AI/JSON Error: {e}")
        traceback.print_exc()
        # Clean up files on error
        for f in files_to_send:
            try:
                genai.delete_file(f.name)
            except:
                pass
        return json.dumps({"visual_fields": [], "confirmation_message": f"Error processing form logic: {str(e)[:100]}"})

    # --- STEP 4: TYPESETTER ENGINE ---
    if doc_path:
        try:
            img = Image.open(doc_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            w, h = img.size
            
            def load_font(size):
                # Define font paths
                kannada_font = os.path.join(BASE_DIR, "NotoSansKannada-Regular.ttf")
                roboto_font = os.path.join(BASE_DIR, "Roboto-Regular.ttf")
                
                selected_font_path = roboto_font # Default to Roboto
                
                # 1. Logic: Select Font based on Language
                # NotoSansKannada contains glyphs for BOTH Kannada and English, so it's safe for mixed text.
                if language == 'kn' and os.path.exists(kannada_font):
                    selected_font_path = kannada_font
                elif os.path.exists(roboto_font):
                    selected_font_path = roboto_font
                else:
                    # Fallback to Arial if Roboto is missing
                    selected_font_path = os.path.join(BASE_DIR, "arial.ttf")

                # 2. Load the selected font
                if os.path.exists(selected_font_path):
                    try: 
                        return ImageFont.truetype(selected_font_path, int(size))
                    except Exception: 
                        pass
                
                # 3. Ultimate Fallback
                return ImageFont.load_default()


            # A. DRAW TEXT FIELDS
            standard_size = int(h * 0.013) 
            standard_size = max(12, min(standard_size, 30)) 
            text_font = load_font(standard_size)

            # A. DRAW TEXT FIELDS (Dynamic Sizing & Alignment Fix)
            # A. DRAW TEXT FIELDS (Bigger Font & Offset Fix)
            # A. DRAW TEXT FIELDS (Smart "Stay Inside Line" Logic)
            for field in data["visual_fields"]:
                vy1, vx1, vy2, vx2 = field.get("value_rect", [0,0,0,0])
                
                # Convert coordinates to pixels
                box_x1 = (vx1 * w) / 1000
                box_x2 = (vx2 * w) / 1000  # <--- CRITICAL: The end of the line
                box_y1 = (vy1 * h) / 1000
                box_y2 = (vy2 * h) / 1000 
                
                # 1. Initial Height-Based Sizing
                box_height = box_y2 - box_y1
                calc_size = int(box_height * 0.80) 
                target_size = max(14, min(calc_size, 65))
                current_font = load_font(target_size)

                # 2. Safety Padding (Gap after label)
                safety_padding = int(w * 0.02) 
                draw_x = box_x1 + safety_padding
                
                # 3. WIDTH CONSTRAINT LOGIC
                # Allowable width = (End of line) - (Start of Text) - (Small Buffer)
                max_text_width = box_x2 - draw_x - 5 
                
                text_val = str(field["value"])
                
                # SHRINK LOOP: If text is wider than the line, reduce font size
                while target_size > 8:
                    # Measure text width with current font
                    text_len = current_font.getlength(text_val) if hasattr(current_font, 'getlength') else current_font.getsize(text_val)[0]
                    
                    if text_len <= max_text_width:
                        break # It fits! Stop shrinking.
                    
                    # Too wide? Shrink and re-measure
                    target_size -= 2
                    current_font = load_font(target_size)

                # 4. Draw (Baseline Aligned)
                draw_y = box_y2 - (target_size * 0.2)
                draw.text((draw_x, draw_y), text_val, fill=(15, 15, 25), font=current_font, anchor="ls")

            # B. DRAW CHECKBOXES (THE "STAY INSIDE" FIX)
            for box in data.get("checkbox_fields", []):
                cy1, cx1, cy2, cx2 = box.get("value_rect", [0,0,0,0])
                
                # 1. Raw Coordinates
                chk_x1 = (cx1 * w) / 1000
                chk_y1 = (cy1 * h) / 1000
                chk_x2 = (cx2 * w) / 1000
                chk_y2 = (cy2 * h) / 1000
                
                raw_w = chk_x2 - chk_x1
                raw_h = chk_y2 - chk_y1
                
                if raw_w > 10 and raw_h > 10:
                    # --- FIX: THE SAFETY INSET ---
                    # We shrink the drawing area by 25% on all sides.
                    # This guarantees the tick stays INSIDE even if the box is small.
                    inset_factor = 0.25 
                    
                    safe_x1 = chk_x1 + (raw_w * inset_factor)
                    safe_y1 = chk_y1 + (raw_h * inset_factor)
                    safe_w = raw_w * (1 - 2 * inset_factor)
                    safe_h = raw_h * (1 - 2 * inset_factor)

                    # --- MANUAL TICK SHAPE (Relative to Safe Zone) ---
                    # P1: Start of short stroke (Left-ish)
                    p1 = (safe_x1 + safe_w * 0.1, safe_y1 + safe_h * 0.55)
                    # P2: The Pivot/Bottom (Center-Bottom)
                    p2 = (safe_x1 + safe_w * 0.4, safe_y1 + safe_h * 0.9)
                    # P3: The End/Top (Top-Right)
                    p3 = (safe_x1 + safe_w * 1.0, safe_y1 + safe_h * 0.0)

                    # Dynamic Thickness (thinner for precision)
                    thickness = max(2, int(min(raw_w, raw_h) * 0.1))

                    # Draw Pure Black Tick
                    draw.line([p1, p2], fill=(0, 0, 0), width=thickness)
                    draw.line([p2, p3], fill=(0, 0, 0), width=thickness)

            output_filename = f"filled_{int(time.time())}.jpg"
            # Use slightly lower quality for faster processing (85 instead of 95)
            img.save(os.path.join(UPLOAD_FOLDER, output_filename), quality=85, optimize=True)
            data["filled_image_url"] = f"/uploads/{output_filename}"
            
        except Exception as e:
            print(f"Render Error: {e}")

    # Clean up uploaded files
    for f in files_to_send:
        try:
            genai.delete_file(f.name)
        except:
            pass
    
    total_form_time = time.time() - form_start
    print(f"✅ [FORM] Total form processing time: {total_form_time:.1f}s")
    print(f"{'='*60}\n")
    
    return json.dumps(data)
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)
@app.route('/analyze_medicine', methods=['POST'])
def analyze_medicine():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    scan_type = request.form.get('scan_type', 'medicine') # Default to medicine
    user_prompt = request.form.get('user_prompt', '')
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(f"vision_{int(time.time())}{pathlib.Path(file.filename).suffix}")
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        return analyze_vision_with_gemini(filepath, scan_type, user_prompt)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)
@app.route('/analyze_vision', methods=['POST'])
def analyze_vision():
    request_start = time.time()
    print(f"\n{'='*60}")
    print(f"🖼️ [VISION] New vision analysis request")
    print(f"{'='*60}")
    
    lang = request.form.get('language', 'en')
    files = request.files.getlist('images') or ([request.files['image']] if 'image' in request.files else [])
    
    paths = []
    for f in files:
        path = os.path.join(UPLOAD_FOLDER, secure_filename(f"v_{int(time.time())}_{f.filename}"))
        f.save(path); paths.append(path)

    try:
        res_str = analyze_vision_with_gemini(paths, request.form.get('scan_type'), request.form.get('user_prompt'), 'en')
        result = json.loads(res_str)
        
        total_time = time.time() - request_start
        print(f"✅ [VISION] Request completed in {total_time:.1f}s")
        print(f"{'='*60}\n")
        
        return jsonify(amazon_translate_dict(result, lang))
    except ValueError as e:
        error_msg = str(e)
        print(f"❌ [VISION] Error: {error_msg}")
        total_time = time.time() - request_start
        print(f"❌ [VISION] Request failed after {total_time:.1f}s")
        print(f"{'='*60}\n")
        
        if "safety" in error_msg.lower() or "blocked" in error_msg.lower():
            return jsonify({
                "error": "Content was blocked by safety filters",
                "answer_to_user": "The image content triggered safety filters. Please try with a different image.",
                "recommendation": "Use a different image or ensure the image is appropriate for medical analysis."
            }), 200
        
        return jsonify({"error": error_msg}), 500
    except Exception as e:
        print(f"❌ [VISION] Error: {e}")
        traceback.print_exc()
        total_time = time.time() - request_start
        print(f"❌ [VISION] Request failed after {total_time:.1f}s")
        print(f"{'='*60}\n")
        return jsonify({"error": str(e)}), 500
    finally:
        for p in paths: 
            if os.path.exists(p): os.remove(p)            
            
def analyze_audio_with_gemini(audio_path, user_prompt, language='en'):
    lang_map = {"en": "English", "hi": "Hindi", "kn": "Kannada"}
    target_lang = lang_map.get(language, "English")

    audio_file = None
    max_retries = 2  # Reduced retries but longer timeouts
    retry_delay = 2
    
    print(f"🎵 [AUDIO] Starting analysis for {target_lang}...")
    start_time = time.time()
    
    try:
        # Upload audio file with extended timeout
        print(f"📤 [AUDIO] Uploading file: {audio_path}")
        audio_file = genai.upload_file(path=audio_path)
        upload_timeout = 60  # Extended to 60 seconds
        upload_start = time.time()
        
        while audio_file.state.name == "PROCESSING":
            elapsed = time.time() - upload_start
            if elapsed > upload_timeout:
                raise TimeoutError(f"Audio upload timeout after {elapsed:.1f}s")
            if int(elapsed) % 5 == 0:  # Log every 5 seconds
                print(f"⏳ [AUDIO] Upload in progress... {elapsed:.1f}s")
            time.sleep(0.5)  # Check every 0.5s
            audio_file = genai.get_file(audio_file.name)
        
        upload_elapsed = time.time() - upload_start
        print(f"✅ [AUDIO] Upload complete in {upload_elapsed:.1f}s")

        safety_config = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE # <--- CHANGED
        }
        
        # Use Pro model for best accuracy - it's more reliable for medical analysis
        print(f"🤖 [AUDIO] Using {MODEL_NAME} for accurate medical analysis...")
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,  # Use pro for accuracy
            system_instruction=SYSTEM_INSTRUCTION,
            safety_settings=safety_config
        )
        
        # Enhanced clinical prompt for accurate analysis
        prompt = f"""Analyze this medical audio recording for respiratory condition assessment.

CLINICAL CONTEXT:
- Patient symptoms/context: {user_prompt if user_prompt else 'General respiratory assessment'}
- This is a legitimate medical audio recording containing breathing sounds, coughs, or respiratory patterns
- Purpose: Clinical evaluation and respiratory condition identification

ANALYSIS REQUIREMENTS:
1. Identify the specific respiratory condition based on audio patterns
2. Describe the exact sounds heard (wheezing, stridor, crackles, cough type, etc.)
3. Provide accurate medical assessment
4. Output language: {target_lang}

Return structured JSON analysis with accurate medical findings."""

        # Retry logic with model fallback if safety filters trigger
        response = None
        response_text = None
        safety_blocked = False
        models_to_try = [MODEL_NAME, "gemini-3-flash-preview"]  # Try pro first, then flash
        current_model_idx = 0
        
        for attempt in range(max_retries * 2):  # Allow trying both models
            try:
                attempt_start = time.time()
                current_model = models_to_try[current_model_idx] if current_model_idx < len(models_to_try) else MODEL_NAME
                print(f"🔄 [AUDIO] Attempt {attempt + 1} - Using {current_model}...")
                
                # Create model if we switched
                if current_model != model._model_name if hasattr(model, '_model_name') else MODEL_NAME:
                    model = genai.GenerativeModel(
                        model_name=current_model,
                        system_instruction=SYSTEM_INSTRUCTION,
                        safety_settings=safety_config
                    )
                
                # Extended timeout: 180 seconds (3 minutes) per attempt
                response = model.generate_content(
                    [audio_file, prompt],
                    generation_config={
                        "response_mime_type": "application/json",
                        "temperature": 0.2,
                        "max_output_tokens": 600  # Reduced further for speed
                    },
                    safety_settings=safety_config,  # Pass safety settings in generate_content too
                    request_options={"timeout": 180}  # 3 minutes per attempt
                )
                
                attempt_elapsed = time.time() - attempt_start
                print(f"✅ [AUDIO] Attempt {attempt + 1} completed in {attempt_elapsed:.1f}s")
                
                # Verify response is valid using safe extraction (handles safety filters)
                try:
                    response_text = safe_get_response_text(response)
                    if response_text:
                        # Success - break out of retry loop
                        safety_blocked = False
                        break
                    else:
                        raise ValueError("Empty response from API")
                except ValueError as ve:
                    # Check if it's a safety filter error
                    error_msg = str(ve).lower()
                    if "safety" in error_msg or "blocked" in error_msg or "recitation" in error_msg:
                        print(f"🚫 [AUDIO] Content blocked by safety filters on attempt {attempt + 1}")
                        safety_blocked = True
                        
                        # Try next model if available
                        if current_model_idx < len(models_to_try) - 1:
                            current_model_idx += 1
                            print(f"🔄 [AUDIO] Switching to {models_to_try[current_model_idx]} as fallback...")
                            continue  # Retry with different model
                        else:
                            # All models blocked - will return default after loop
                            break
                    else:
                        raise
                
            except ValueError as ve:
                # Handle ValueError separately (safety filters)
                error_str = str(ve).lower()
                if "safety" in error_str or "blocked" in error_str or "recitation" in error_str:
                    print(f"🚫 [AUDIO] Safety filter ValueError on attempt {attempt + 1}")
                    safety_blocked = True
                    # Try next model if available
                    if current_model_idx < len(models_to_try) - 1:
                        current_model_idx += 1
                        print(f"🔄 [AUDIO] Switching to {models_to_try[current_model_idx]}...")
                        continue
                    else:
                        break  # Exit loop to return default
                else:
                    raise  # Re-raise other ValueErrors
                
            except Exception as e:
                error_str = str(e).lower()
                attempt_elapsed = time.time() - attempt_start
                
                # Check if it's a safety filter error
                if "safety" in error_str or "blocked" in error_str or "recitation" in error_str or "finish_reason" in error_str:
                    print(f"🚫 [AUDIO] Safety filter triggered on attempt {attempt + 1}")
                    safety_blocked = True
                    # Try next model if available
                    if current_model_idx < len(models_to_try) - 1:
                        current_model_idx += 1
                        print(f"🔄 [AUDIO] Switching to {models_to_try[current_model_idx]}...")
                        continue
                    else:
                        break  # Exit loop to return default
                
                # Check if it's a timeout/deadline error
                if "deadline" in error_str or "timeout" in error_str or "504" in error_str:
                    print(f"⏱️ [AUDIO] Timeout on attempt {attempt + 1} after {attempt_elapsed:.1f}s")
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                        print(f"⏳ [AUDIO] Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        # Last attempt failed
                        total_elapsed = time.time() - start_time
                        raise TimeoutError(f"Audio analysis timed out after {max_retries} attempts ({total_elapsed:.1f}s total)")
                else:
                    # Non-timeout error - log and re-raise
                    print(f"❌ [AUDIO] Error on attempt {attempt + 1}: {str(e)}")
                    raise
        
        # Check if we need to return default response due to safety filters
        if safety_blocked and not response_text:
            print(f"⚠️ [AUDIO] All attempts blocked by safety filters. Returning default analysis response...")
            # Clean up before returning
            if audio_file:
                try:
                    genai.delete_file(audio_file.name)
                except:
                    pass
            # Return a safe default response
            default_response = {
                "valid_audio": True,
                "universal_match": {"disease_name": "Unable to Analyze", "similarity_score": 0},
                "severity": "Unknown",
                "infection_type": "Analysis Blocked",
                "simple_explanation": "The audio analysis was blocked by content filters. This may occur with unclear audio, background noise, or other factors. Please try recording again in a quiet environment with clear speech.",
                "audio_characteristics": "Analysis unavailable due to content filtering.",
                "recommendation": "Please record a new audio sample in a quiet environment. Ensure the recording is clear and contains only the respiratory sounds you want analyzed."
            }
            total_elapsed = time.time() - start_time
            print(f"✅ [AUDIO] Default response returned after {total_elapsed:.1f}s")
            return json.dumps(default_response)
        
        # Verify we have a valid response and response text
        if not response or not response_text:
            raise ValueError("No valid response received from API")
        
        print(f"📝 [AUDIO] Raw response length: {len(response_text)} chars")
        
        total_elapsed = time.time() - start_time
        print(f"✅ [AUDIO] Analysis complete in {total_elapsed:.1f}s total")
        
        # Clean up uploaded file
        if audio_file:
            try:
                genai.delete_file(audio_file.name)
                print(f"🗑️ [AUDIO] Cleaned up uploaded file")
            except:
                pass  # Ignore cleanup errors
        
        # Process response with robust JSON parsing
        print(f"📊 [AUDIO] Processing response...")
        
        # Try to parse JSON with cleaning
        data = None
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"⚠️ [AUDIO] JSON parse error, attempting to clean: {e}")
            data = clean_json_response(response_text)
            
            if data is None:
                # Last attempt: try to extract just the JSON part
                print(f"⚠️ [AUDIO] Cleaning failed, attempting manual extraction...")
                # Try to find and extract JSON object
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    try:
                        data = json.loads(json_match.group(0))
                    except:
                        pass
                
                if data is None:
                    raise ValueError(f"Could not parse JSON from response. Error: {e}")
        
        print(f"✅ [AUDIO] JSON parsed successfully")
        
        # FIX: Handle case where Gemini returns a list instead of a dict
        if isinstance(data, list):
            data = data[0] if len(data) > 0 else {}

        # --- CRITICAL FIX: Safety Check for Null Data ---
        if not data: 
            data = {}

        match = data.get("universal_match")
        # If 'universal_match' is missing OR explicitly null, use empty dict
        if not match: 
            match = {}

        score = int(match.get("similarity_score", 0))
        disease = match.get("disease_name", "Unknown Condition")
        
        final_condition = ""
        risk_label = ""

        # --- ADJUSTED THRESHOLDS ---
        if score > 70:
            final_condition = disease
            risk_label = f"{data.get('infection_type', 'Condition')} ({data.get('severity', 'Moderate')} Risk)"
        elif score >= 40:
            final_condition = "Respiratory Irritation"
            risk_label = "General Observation (Low Risk)"
        else:
            final_condition = "Unclear Cough Pattern"
            risk_label = "Inconclusive Analysis"

        # Sanitize Strings
        simple_expl = data.get("simple_explanation", "Analysis complete.")
        if simple_expl:
            simple_expl = simple_expl.replace('"', '').replace("'", "")

        formatted_output = {
            "valid_audio": data.get("valid_audio", True),
            "condition": final_condition,       
            "disease_type": risk_label,         
            "severity": data.get("severity", "Moderate"),
            "acoustic_analysis": data.get("audio_characteristics", "No specific patterns detected."), 
            "simple_explanation": simple_expl,
            "recommendation": data.get("recommendation", "Please consult a doctor.")
        }
        
        return json.dumps(formatted_output)

    except ValueError as e:
        # Handle safety filter and other value errors
        error_msg = str(e).lower()
        total_elapsed = time.time() - start_time if 'start_time' in locals() else 0
        
        if "safety" in error_msg or "blocked" in error_msg or "recitation" in error_msg:
            print(f"🚫 [AUDIO] Safety filter triggered after {total_elapsed:.1f}s")
        else:
            print(f"❌ [AUDIO] ValueError after {total_elapsed:.1f}s: {e}")
        
        # Clean up on error
        if audio_file:
            try:
                genai.delete_file(audio_file.name)
            except:
                pass
        
        if "safety" in error_msg or "blocked" in error_msg:
            return json.dumps({
                "valid_audio": True,
                "condition": "Content Filtered",
                "disease_type": "Safety Filter",
                "simple_explanation": "The audio content was blocked by safety filters. This may happen if the recording contains background noise, unclear speech, or other content that triggered filters.",
                "recommendation": "Please try recording again with clearer speech in a quiet environment. Ensure the recording contains only the respiratory sounds you want analyzed.",
                "acoustic_analysis": "N/A",
                "severity": "Unknown"
            })
        else:
            return json.dumps({
                "valid_audio": True,
                "condition": "Analysis Error",
                "disease_type": "System Error",
                "simple_explanation": f"Could not process audio data. {str(e)[:150]}",
                "recommendation": "Please try again with a shorter audio clip or check your connection.",
                "acoustic_analysis": "N/A",
                "severity": "Unknown"
            })
    except TimeoutError as e:
        total_elapsed = time.time() - start_time if 'start_time' in locals() else 0
        print(f"⏱️ [AUDIO] Timeout Error after {total_elapsed:.1f}s: {e}")
        # Clean up on timeout
        if audio_file:
            try:
                genai.delete_file(audio_file.name)
            except:
                pass
        return json.dumps({
            "valid_audio": True,
            "condition": "Processing Timeout",
            "disease_type": "System Timeout",
            "simple_explanation": f"The analysis timed out after {total_elapsed:.0f} seconds. The audio file may be too long or the connection is slow.",
            "recommendation": "Try recording a shorter audio sample (under 30 seconds) or check your internet connection.",
            "acoustic_analysis": "N/A",
            "severity": "Unknown"
        })
    except Exception as e:
        total_elapsed = time.time() - start_time if 'start_time' in locals() else 0
        print(f"❌ [AUDIO] Error after {total_elapsed:.1f}s: {e}")
        traceback.print_exc()  # Full stack trace for debugging
        # Clean up on any error
        if audio_file:
            try:
                genai.delete_file(audio_file.name)
            except:
                pass
        return json.dumps({
            "valid_audio": True,
            "condition": "Analysis Error",
            "disease_type": "System Error",
            "simple_explanation": f"Could not process audio data. Error: {str(e)[:100]}",
            "recommendation": "Please try again with a shorter audio clip or check your connection.",
            "acoustic_analysis": "N/A",
            "severity": "Unknown"
        })    
@app.route('/')

def index():
    return render_template('index.html')

translate_client = boto3.client(
    service_name='translate', 
    region_name=os.getenv("AWS_REGION", "us-east-1"), 
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    use_ssl=True
)
@app.route('/translate_ui_bulk', methods=['POST'])
def translate_ui_bulk():
    data = request.json
    texts = data.get('texts', [])
    target_lang = data.get('target_lang', 'en')
    
    if not texts or target_lang == 'en':
        return jsonify({"translated_texts": texts})

    translated_list = []
    try:
        for text in texts:
            if text.strip() == "" or text.isdigit():
                translated_list.append(text)
                continue
            
            result = translate_client.translate_text(
                Text=text, 
                SourceLanguageCode="en", 
                TargetLanguageCode=target_lang
            )
            translated_list.append(result.get('TranslatedText'))
        
        return jsonify({"translated_texts": translated_list})
    except Exception as e:
        print(f"AWS Error: {e}")
        return jsonify({"translated_texts": texts, "error": str(e)})
def amazon_translate_dict(data, target_lang):
    """Recursively translates all string values in a dictionary/list."""
    if not target_lang or target_lang == 'en':
        return data

    if isinstance(data, dict):
        return {k: amazon_translate_dict(v, target_lang) for k, v in data.items()}
    elif isinstance(data, list):
        return [amazon_translate_dict(item, target_lang) for item in data]
    elif isinstance(data, str):
        try:
            # Use 'auto' to allow Amazon to detect source language if needed
            result = translate_client.translate_text(
                Text=data, SourceLanguageCode="auto", TargetLanguageCode=target_lang
            )
            return result.get('TranslatedText')
        except Exception as e:
            print(f"Translation error: {e}")
            return data
    return data


@app.route('/analyze', methods=['POST'])
def analyze():
    request_start = time.time()
    print(f"\n{'='*60}")
    print(f"🎤 [REQUEST] New audio analysis request received")
    print(f"{'='*60}")
    
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file"}), 400
    
    lang = request.form.get('language', 'en')
    file = request.files['audio']
    
    # Get file size for logging
    file.seek(0, 2)  # Seek to end
    file_size = file.tell()
    file.seek(0)  # Reset
    
    print(f"📁 [REQUEST] File size: {file_size / 1024:.1f} KB")
    print(f"🌐 [REQUEST] Language: {lang}")
    
    # Support both webm and wav
    ext = 'webm' if file.filename and 'webm' in file.filename.lower() else 'wav'
    filepath = os.path.join(UPLOAD_FOLDER, secure_filename(f"t_{int(time.time())}.{ext}"))
    file.save(filepath)

    try:
        # 1. Analyze in English (highest accuracy)
        res_str = analyze_audio_with_gemini(filepath, request.form.get('user_prompt', ''), 'en')
        # 2. Translate everything via Amazon
        result = json.loads(res_str)
        
        total_time = time.time() - request_start
        print(f"✅ [REQUEST] Request completed in {total_time:.1f}s")
        print(f"{'='*60}\n")
        
        return jsonify(amazon_translate_dict(result, lang))
    except ValueError as e:
        error_msg = str(e)
        print(f"❌ [REQUEST] Value Error: {error_msg}")
        total_time = time.time() - request_start
        print(f"❌ [REQUEST] Request failed after {total_time:.1f}s")
        print(f"{'='*60}\n")
        
        # Handle safety filter errors
        if "safety" in error_msg.lower() or "blocked" in error_msg.lower():
            return jsonify({
                "valid_audio": True,
                "condition": "Content Filtered",
                "disease_type": "Safety Filter",
                "simple_explanation": "The audio content triggered safety filters. Please try with a different recording.",
                "recommendation": "Record a new audio sample with clearer speech.",
                "acoustic_analysis": "N/A",
                "severity": "Unknown"
            }), 200  # Return 200 so frontend can display the message
        
        return jsonify({
            "valid_audio": True,
            "condition": "Processing Error",
            "disease_type": "System Error",
            "simple_explanation": f"Analysis failed: {error_msg[:100]}",
            "recommendation": "Please try again with a shorter audio clip.",
            "acoustic_analysis": "N/A",
            "severity": "Unknown"
        }), 500
    except json.JSONDecodeError as e:
        print(f"❌ [REQUEST] JSON Decode Error: {e}")
        total_time = time.time() - request_start
        print(f"❌ [REQUEST] Request failed after {total_time:.1f}s")
        print(f"{'='*60}\n")
        return jsonify({
            "valid_audio": True,
            "condition": "Processing Error",
            "disease_type": "Format Error",
            "simple_explanation": "Could not parse the analysis result. The response format was invalid.",
            "recommendation": "Retry with a shorter audio clip or try again.",
            "acoustic_analysis": "N/A",
            "severity": "Unknown"
        }), 500
    except Exception as e:
        print(f"❌ [REQUEST] Analyze Route Error: {e}")
        traceback.print_exc()
        total_time = time.time() - request_start
        print(f"❌ [REQUEST] Request failed after {total_time:.1f}s")
        print(f"{'='*60}\n")
        return jsonify({
            "valid_audio": True,
            "condition": "System Error",
            "disease_type": "Error",
            "simple_explanation": f"An error occurred during analysis: {str(e)[:100]}",
            "recommendation": "Check your connection and retry.",
            "acoustic_analysis": "N/A",
            "severity": "Unknown"
        }), 500
    finally:
        if os.path.exists(filepath): 
            try:
                os.remove(filepath)
            except:
                pass        
@app.route('/process_form_voice', methods=['POST'])
def process_form_voice():
    lang = request.form.get('language', 'en')
    audio = request.files.get('audio')
    doc = request.files.get('form_doc')
    
    a_path = os.path.join(UPLOAD_FOLDER, "v.wav") if audio else None
    if a_path: audio.save(a_path)
    d_path = os.path.join(UPLOAD_FOLDER, doc.filename) if doc else None
    if d_path: doc.save(d_path)

    try:
        res_str = analyze_form_voice(a_path, request.form.get('text_input'), request.form.get('mode'), d_path, 'en')
        return jsonify(amazon_translate_dict(json.loads(res_str), lang))
    finally:
        if a_path and os.path.exists(a_path): os.remove(a_path)        
        
if __name__ == '__main__':
    # Use the port assigned by the server, default to 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)