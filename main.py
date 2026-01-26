import os
import time
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from werkzeug.utils import secure_filename
from datetime import datetime
import json
import pathlib
import boto3
import numpy as np # Ensure numpy is imported
import os
from PIL import Image, ImageDraw, ImageFont
from flask import send_from_directory
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
import urllib.request
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
import uuid # <--- ADD THIS AT THE TOP WITH OTHER IMPORTS
# --- UPDATED SYSTEM INSTRUCTION (UNIVERSAL PATHOLOGICAL MATCHER) ---
# --- UPDATED SYSTEM INSTRUCTION (SIMPLE & PATIENT-FRIENDLY) ---
SYSTEM_INSTRUCTION = """
You are Sahayak.ai, a compassionate medical AI assistant.
Your task is to analyze audio and identify respiratory conditions using Universal Vector Matching.

### ANALYSIS PROTOCOL:
1. **Listen & Match**: Compare the audio to your internal database of disease sounds (Croup, Asthma, Pneumonia, etc.).
2. **Internal Scoring**: Calculate a match confidence (0-100) internally to ensure accuracy, but **DO NOT** show this number to the user.
3. **Simplify**: Translate all findings into simple, easy-to-understand language for a non-medical user.

### OUTPUT FORMAT (Strict JSON):
{
  "valid_audio": true,
  "universal_match": {
    "disease_name": "Medical Name (e.g., Croup)",
    "similarity_score": 95
  },
  "severity": "Moderate / High / Low",
  "infection_type": "Viral / Bacterial / Chronic / Irritation",
  "simple_explanation": "A direct, clear explanation of what this condition is. Do NOT use quotes. Do NOT mention percentages. Example: 'This sounds like Croup, which is usually caused by a virus and causes swelling in the throat.'",
  "audio_characteristics": "What did you hear? Explain in plain English. Example: 'We detected a distinctive barking sound, similar to a seal, and some whistling noises when breathing in.'",
  "recommendation": "Simple, actionable advice. Example: 'Keep the patient calm and try sitting in a steamy bathroom to help them breathe.'"
}
"""

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
            # Optimized: Check file state with timeout and refresh
            max_wait = 60
            wait_time = 0
            while img_file.state.name == "PROCESSING" and wait_time < max_wait:
                time.sleep(0.5)  # Reduced from 1 second
                img_file = genai.get_file(img_file.name)
                wait_time += 0.5
            
            if img_file.state.name != "ACTIVE":
                print(f"Image upload failed or timed out: {img_file.state.name}")
                continue  # Skip this image
            
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

    model = genai.GenerativeModel(model_name=MODEL_NAME, system_instruction=sys_instruction)
    
    response = model.generate_content(
        content_payload,
        generation_config={"response_mime_type": "application/json"}
    )
    
    for ref in uploaded_refs:
        genai.delete_file(ref.name)

    # Fix: Check if response has valid parts before accessing .text
    if response.candidates and len(response.candidates) > 0:
        candidate = response.candidates[0]
        if candidate.content and candidate.content.parts:
            return candidate.content.parts[0].text.strip()
        else:
            print("Vision Analysis Error: Response has no valid parts")
            return json.dumps({"error": "AI response has no valid content"})
    else:
        print("Vision Analysis Error: Response has no candidates")
        return json.dumps({"error": "AI response failed"})


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

    # --- STEP 1: TRANSCRIBE AUDIO ---
    final_text_input = text_input or ""
    print(f"Initial text input: '{final_text_input}'")
    
    if audio_path:
        try:
            audio_file = genai.upload_file(path=audio_path)
            # Optimized: Check file state with timeout and refresh
            max_wait = 30  # 30 second timeout
            wait_time = 0
            while audio_file.state.name == "PROCESSING" and wait_time < max_wait:
                time.sleep(0.5)  # Reduced from 1 second
                audio_file = genai.get_file(audio_file.name)
                wait_time += 0.5
            
            if audio_file.state.name != "ACTIVE":
                print(f"Audio upload failed or timed out: {audio_file.state.name}")
                raise Exception("Audio upload failed")
            
            # Use a more reliable model for transcription
            transcribe_model = genai.GenerativeModel("gemini-1.5-flash")  # More reliable for transcription
            
            # Improved transcription prompt - more direct
            transcribe_prompt = f"""You are a speech-to-text transcription system. Listen to this audio and write down EXACTLY what the person says.

IMPORTANT:
- Write down every single word you hear
- Include all names, numbers, dates, addresses
- Keep the exact words and phrases
- Do NOT add punctuation unless you hear pauses
- Do NOT summarize or interpret
- If you hear "Name is John Smith", write "Name is John Smith"
- If you hear "Age is 25 years", write "Age is 25 years"
- If you hear "Date is January 27", write "Date is January 27"

Output ONLY the transcribed text in {target_lang}. No explanations, no quotes, just the words:"""
            
            print(f"Starting transcription with audio file: {audio_file.name}")
            transcribe_res = transcribe_model.generate_content(
                [audio_file, transcribe_prompt],
                generation_config={
                    "temperature": 0.1,  # Lower temperature for more accurate transcription
                    "max_output_tokens": 2000,  # Increased for longer speech
                    "top_p": 0.95,
                    "top_k": 40
                }
            )
            print(f"Transcription response received, checking candidates...")
            
            # Fix: Check if response has valid parts before accessing .text
            if transcribe_res.candidates and len(transcribe_res.candidates) > 0:
                candidate = transcribe_res.candidates[0]
                if candidate.content and candidate.content.parts:
                    transcribed_text = candidate.content.parts[0].text.strip()
                    
                    # Validate transcription - reject if it's just punctuation or too short
                    if transcribed_text and len(transcribed_text) > 1 and transcribed_text not in [".", ",", "!", "?", "...", "no speech detected", "No speech detected", "No speech", "no speech"]:
                        final_text_input += f" {transcribed_text}"
                        print(f"✓ Transcribed text: '{transcribed_text}'")
                    else:
                        print(f"✗ Transcription appears invalid or empty: '{transcribed_text}'")
                        if not final_text_input.strip():
                            print("WARNING: No valid transcription and no text input provided!")
                            # Try alternative transcription method
                            try:
                                print("Attempting alternative transcription...")
                                alt_transcribe = transcribe_model.generate_content(
                                    [audio_file, "What did the person say in this audio? Transcribe every word exactly."],
                                    generation_config={"temperature": 0.3, "max_output_tokens": 1000}
                                )
                                if alt_transcribe.candidates and len(alt_transcribe.candidates) > 0:
                                    alt_candidate = alt_transcribe.candidates[0]
                                    if alt_candidate.content and alt_candidate.content.parts:
                                        alt_text = alt_candidate.content.parts[0].text.strip()
                                        if alt_text and len(alt_text) > 1 and alt_text not in [".", ",", "!", "?", "..."]:
                                            final_text_input = alt_text
                                            print(f"✓ Alternative transcription successful: '{alt_text}'")
                            except Exception as alt_e:
                                print(f"Alternative transcription also failed: {alt_e}")
                else:
                    print("Transcription Error: Response has no valid parts")
            else:
                print("Transcription Error: Response has no candidates")
            
            genai.delete_file(audio_file.name)
            
        except Exception as e:
            print(f"Transcription Error: {e}")

    # --- STEP 2: PREPARE IMAGE ---
    if doc_path:
        print(f"Preparing document: {doc_path}")
        doc_file = genai.upload_file(path=doc_path)
        # Optimized: Check file state with timeout and refresh
        max_wait = 60  # 60 second timeout for larger files
        wait_time = 0
        while doc_file.state.name == "PROCESSING" and wait_time < max_wait:
            time.sleep(0.5)  # Reduced from 1 second
            doc_file = genai.get_file(doc_file.name)
            wait_time += 0.5
        
        if doc_file.state.name != "ACTIVE":
            print(f"Document upload failed or timed out: {doc_file.state.name}")
            raise Exception("Document upload failed")
        
        files_to_send.append(doc_file)
        print(f"Document uploaded successfully")
    
    print(f"Final text input for AI: '{final_text_input}'")
    
    # Validate that we have meaningful input
    if not final_text_input or len(final_text_input.strip()) < 3:
        print("WARNING: Final text input is too short or empty. This may result in no fields being extracted.")
        if doc_path:
            return json.dumps({
                "visual_fields": [], 
                "checkbox_fields": [],
                "confirmation_message": "Error: No valid speech or text input detected. Please speak clearly or type your instructions."
            })

    # --- STEP 3: LOGIC & ALIGNMENT ---
    # Enhanced prompt for better field extraction - more aggressive extraction
    EXTREME_PROMPT = f"""You are an expert document form filler. Your job is to fill out the form in the image based on what the user said.

USER SAID: "{final_text_input}"

STEP 1: Look at the document image. Identify every form field you see (Name, Age, Date, Address, Phone, Email, etc. and any checkboxes).

STEP 2: Analyze what the user said: "{final_text_input}"
- Extract ANY names, numbers, dates, addresses, or other information mentioned
- Even if the transcription is short, try to extract what you can
- Look for common patterns: "Name is X", "Age is X", "Date is X", "Address is X"
- If user says just "John" or "John Smith", assume it's for the Name field
- If user says "25" or "25 years", assume it's for Age field
- Be smart about matching partial information to form fields

STEP 3: For each piece of information you extracted:
- Find the matching field on the form
- Create a bounding box (value_rect) for where to write the text
- The bounding box should cover the blank space AFTER the field label

CRITICAL:
- Extract EVERYTHING the user mentioned, even if it seems incomplete
- If user said "John", fill the Name field with "John"
- If user said "25", fill the Age field with "25"
- Be proactive - match partial information to likely fields
- Value must be in {target_lang} language
- value_rect format: [ymin, xmin, ymax, xmax] - start xmin AFTER the label, leave 4 spaces gap
- Never use "none", "None", "N/A", or empty values

OUTPUT JSON:
{{
  "visual_fields": [{{"key": "Field Name", "value": "Extracted Value", "value_rect": [ymin, xmin, ymax, xmax]}}],
  "checkbox_fields": [{{"key": "Checkbox Label", "value_rect": [ymin, xmin, ymax, xmax]}}]
}}

IMPORTANT: Extract as much as possible from "{final_text_input}". Even single words or numbers should be matched to appropriate form fields."""

    # Use faster model for document analysis
    model = genai.GenerativeModel(model_name="gemini-2.0-flash-exp")  # Faster than pro
    
    try:
        response = model.generate_content(
            files_to_send + [EXTREME_PROMPT], 
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.2,  # Lower temperature for faster, more deterministic responses
                "max_output_tokens": 4000  # Limit output size for faster processing
            }
        )
        
        # Fix: Check if response has valid parts before accessing .text
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                response_text = candidate.content.parts[0].text.strip()
                data = json.loads(response_text)
            else:
                print("AI Error: Response has no valid parts")
                return json.dumps({"visual_fields": [], "confirmation_message": "Error: AI response has no valid content."})
        else:
            print("AI Error: Response has no candidates")
            return json.dumps({"visual_fields": [], "confirmation_message": "Error: AI response failed."})
        
        if "visual_fields" not in data: data["visual_fields"] = []
        if "checkbox_fields" not in data: data["checkbox_fields"] = []
        
        print(f"AI returned {len(data['visual_fields'])} visual_fields and {len(data['checkbox_fields'])} checkbox_fields")
        
        # Fix: Filter out invalid fields (none, None, empty, etc.)
        invalid_values = ["none", "None", "N/A", "n/a", "null", "Null", ""]
        original_count = len(data["visual_fields"])
        data["visual_fields"] = [
            field for field in data["visual_fields"] 
            if field.get("value") and str(field.get("value")).strip() not in invalid_values
        ]
        filtered_count = len(data["visual_fields"])
        if original_count != filtered_count:
            print(f"Filtered out {original_count - filtered_count} invalid fields. Remaining: {filtered_count}")
        
    except Exception as e:
        print(f"AI/JSON Error: {e}")
        return json.dumps({"visual_fields": [], "confirmation_message": "Error processing form logic."})

    # --- STEP 4: TYPESETTER ENGINE ---
    # Only fill document if we have valid fields to fill
    if doc_path and (len(data.get("visual_fields", [])) > 0 or len(data.get("checkbox_fields", [])) > 0):
        try:
            print(f"Filling document with {len(data.get('visual_fields', []))} text fields and {len(data.get('checkbox_fields', []))} checkboxes")
            img = Image.open(doc_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            w, h = img.size
            
            # Cache fonts to avoid reloading
            _font_cache = {}
            
            def load_font(size):
                cache_key = f"{language}_{int(size)}"
                if cache_key in _font_cache:
                    return _font_cache[cache_key]
                
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
                font_obj = ImageFont.load_default()  # Default fallback
                if os.path.exists(selected_font_path):
                    try: 
                        font_obj = ImageFont.truetype(selected_font_path, int(size))
                    except Exception: 
                        pass
                
                # Cache the font
                _font_cache[cache_key] = font_obj
                return font_obj


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
                
                # SHRINK LOOP: If text is wider than the line, reduce font size (optimized)
                while target_size > 8:
                    # Measure text width with current font (use faster method)
                    try:
                        text_len = current_font.getlength(text_val)
                    except:
                        try:
                            text_len = current_font.getsize(text_val)[0]
                        except:
                            text_len = len(text_val) * (target_size * 0.6)  # Rough estimate
                    
                    if text_len <= max_text_width:
                        break # It fits! Stop shrinking.
                    
                    # Too wide? Shrink and re-measure (larger steps for speed)
                    target_size -= 3  # Increased from 2 to 3 for faster processing
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
            output_path = os.path.join(UPLOAD_FOLDER, output_filename)
            # Optimized: Use quality 85 instead of 95 for faster saving (minimal quality difference)
            img.save(output_path, quality=85, optimize=True)
            data["filled_image_url"] = f"/uploads/{output_filename}"
            print(f"Document filled successfully: {output_filename}")
            
        except Exception as e:
            print(f"Render Error: {e}")
            import traceback
            traceback.print_exc()
            # Don't set filled_image_url if there was an error
            if "filled_image_url" in data:
                del data["filled_image_url"]
    elif doc_path:
        # Document provided but no valid fields to fill
        print(f"Warning: Document provided but no valid fields to fill. visual_fields: {len(data.get('visual_fields', []))}, checkbox_fields: {len(data.get('checkbox_fields', []))}")
        data["confirmation_message"] = "Document uploaded but no valid fields were found to fill. Please check your input."

    for f in files_to_send: genai.delete_file(f.name)
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
    lang = request.form.get('language', 'en')
    files = request.files.getlist('images') or ([request.files['image']] if 'image' in request.files else [])
    
    paths = []
    for f in files:
        path = os.path.join(UPLOAD_FOLDER, secure_filename(f"v_{int(time.time())}_{f.filename}"))
        f.save(path); paths.append(path)

    try:
        res_str = analyze_vision_with_gemini(paths, request.form.get('scan_type'), request.form.get('user_prompt'), 'en')
        return jsonify(amazon_translate_dict(json.loads(res_str), lang))
    finally:
        for p in paths: 
            if os.path.exists(p): os.remove(p)            
            
def analyze_audio_with_gemini(audio_path, user_prompt, language='en'):
    lang_map = {"en": "English", "hi": "Hindi", "kn": "Kannada"}
    target_lang = lang_map.get(language, "English")

    audio_file = genai.upload_file(path=audio_path)
    # Optimized: Check file state with timeout and refresh
    max_wait = 30
    wait_time = 0
    while audio_file.state.name == "PROCESSING" and wait_time < max_wait:
        time.sleep(0.5)  # Reduced from 1 second
        audio_file = genai.get_file(audio_file.name)
        wait_time += 0.5
    
    if audio_file.state.name != "ACTIVE":
        print(f"Audio upload failed or timed out: {audio_file.state.name}")
        raise Exception("Audio upload failed")

    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=SYSTEM_INSTRUCTION
    )
    
    prompt = f"""
    Analyze this audio.
    Context: {user_prompt}
    Target Language: {target_lang}.
    
    CRITICAL INSTRUCTION: 
    1. Output strictly plain text for explanations (NO markdown, NO quotes).
    2. Translate all value strings to {target_lang}.
    """

    response = model.generate_content(
        [audio_file, prompt],
        generation_config={"response_mime_type": "application/json", "temperature": 0.2}
    )

    genai.delete_file(audio_file.name)
    
    try:
        # Fix: Check if response has valid parts before accessing .text
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                response_text = candidate.content.parts[0].text.strip()
                data = json.loads(response_text)
            else:
                print("Audio Analysis Error: Response has no valid parts")
                data = {}
        else:
            print("Audio Analysis Error: Response has no candidates")
            data = {}
        
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

    except Exception as e:
        print(f"Logic Error: {e}")
        return json.dumps({
            "valid_audio": True,
            "condition": "Analysis Error",
            "disease_type": "System Error",
            "simple_explanation": "Could not process audio data safely. Please try again.",
            "recommendation": "Check internet connection.",
            "acoustic_analysis": "N/A"
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
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file"}), 400
    
    lang = request.form.get('language', 'en')
    file = request.files['audio']
    filepath = os.path.join(UPLOAD_FOLDER, secure_filename(f"t_{int(time.time())}.wav"))
    file.save(filepath)

    try:
        # 1. Analyze in English (highest accuracy)
        res_str = analyze_audio_with_gemini(filepath, request.form.get('user_prompt', ''), 'en')
        # 2. Translate everything via Amazon
        return jsonify(amazon_translate_dict(json.loads(res_str), lang))
    finally:
        if os.path.exists(filepath): os.remove(filepath)        
@app.route('/process_form_voice', methods=['POST'])
def process_form_voice():
    lang = request.form.get('language', 'en')
    audio = request.files.get('audio')
    doc = request.files.get('form_doc')
    text_input = request.form.get('text_input', '')
    
    a_path = None
    if audio:
        a_path = os.path.join(UPLOAD_FOLDER, f"v_{int(time.time())}.wav")  # Unique filename
        try:
            audio.save(a_path)
            # Validate audio file exists and has content
            if os.path.exists(a_path):
                file_size = os.path.getsize(a_path)
                print(f"Audio file saved: {a_path}, size: {file_size} bytes")
                if file_size < 1000:  # Less than 1KB is likely invalid
                    print(f"WARNING: Audio file is very small ({file_size} bytes), may be corrupted or empty")
            else:
                print("ERROR: Audio file was not saved properly")
                a_path = None
        except Exception as e:
            print(f"Error saving audio file: {e}")
            a_path = None
    
    d_path = None
    if doc and doc.filename:
        d_path = os.path.join(UPLOAD_FOLDER, secure_filename(doc.filename))
        try:
            doc.save(d_path)
            print(f"Document saved: {d_path}")
        except Exception as e:
            print(f"Error saving document: {e}")
            d_path = None

    try:
        print(f"Processing with audio: {a_path is not None}, text_input: '{text_input}', doc: {d_path is not None}")
        res_str = analyze_form_voice(a_path, text_input, request.form.get('mode'), d_path, lang)
        return jsonify(amazon_translate_dict(json.loads(res_str), lang))
    except Exception as e:
        print(f"Error in process_form_voice: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "visual_fields": [],
            "checkbox_fields": [],
            "confirmation_message": f"Error processing request: {str(e)}"
        }), 500
    finally:
        if a_path and os.path.exists(a_path): 
            os.remove(a_path)
        if d_path and os.path.exists(d_path): 
            os.remove(d_path)        
        
if __name__ == '__main__':
    # Use the port assigned by the server, default to 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)