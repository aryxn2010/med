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
SYSTEM_INSTRUCTION = """You are Sahayak.ai, a medical AI assistant. Analyze audio to identify respiratory conditions.

PROTOCOL: Match audio to disease sounds (Croup, Asthma, Pneumonia). Calculate confidence (0-100) internally. Use simple language.

OUTPUT JSON:
{
  "valid_audio": true,
  "universal_match": {"disease_name": "Condition Name", "similarity_score": 95},
  "severity": "Moderate/High/Low",
  "infection_type": "Viral/Bacterial/Chronic/Irritation",
  "simple_explanation": "Plain explanation without quotes or percentages.",
  "audio_characteristics": "What you heard in plain English.",
  "recommendation": "Actionable advice."
}"""

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

    model = genai.GenerativeModel(model_name=MODEL_NAME, system_instruction=sys_instruction)
    
    try:
        response = model.generate_content(
            content_payload,
            generation_config={
                "response_mime_type": "application/json",
                "max_output_tokens": 1500  # Limit for faster response
            },
            request_options={"timeout": 60}  # 60 second timeout
        )
    except Exception as e:
        # Clean up uploaded files on error
        for ref in uploaded_refs:
            try:
                genai.delete_file(ref.name)
            except:
                pass
        raise  # Re-raise to be handled by caller
    
    for ref in uploaded_refs:
        genai.delete_file(ref.name)

    return response.text.strip()


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
    final_text_input = text_input or ""
    audio_file = None
    
    if audio_path:
        try:
            # Upload audio file with faster checking
            audio_file = genai.upload_file(path=audio_path)
            upload_timeout = 25  # Reduced timeout
            upload_start = time.time()
            
            while audio_file.state.name == "PROCESSING":
                if time.time() - upload_start > upload_timeout:
                    raise TimeoutError("Audio upload timeout")
                time.sleep(0.3)  # Faster checking interval
                audio_file = genai.get_file(audio_file.name)
            
            # Use faster flash model for transcription
            transcribe_model = genai.GenerativeModel("gemini-3-flash-preview")
            
            transcribe_res = transcribe_model.generate_content(
                [audio_file, f"Transcribe this audio exactly into {target_lang}. Return ONLY the text."],
                generation_config={"max_output_tokens": 500},  # Limit tokens for speed
                request_options={"timeout": 30}  # Reduced timeout
            )
            
            transcribed_text = transcribe_res.text.strip()
            final_text_input += f" {transcribed_text}"
            
            # Clean up immediately after transcription
            if audio_file:
                try:
                    genai.delete_file(audio_file.name)
                except:
                    pass
            
        except Exception as e:
            print(f"Transcription Error: {e}")
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

    # Use flash model for faster processing when no document, pro when document needs analysis
    model_name = "gemini-3-flash-preview" if not doc_path else "gemini-3-pro-preview"
    model = genai.GenerativeModel(model_name=model_name)
    
    try:
        response = model.generate_content(
            files_to_send + [EXTREME_PROMPT], 
            generation_config={
                "response_mime_type": "application/json",
                "max_output_tokens": 1500,  # Reduced for faster response
                "temperature": 0.3  # Lower temperature for faster, more deterministic output
            },
            request_options={"timeout": 45}  # Reduced timeout
        )
        data = json.loads(response.text)
        if "visual_fields" not in data: data["visual_fields"] = []
        if "checkbox_fields" not in data: data["checkbox_fields"] = []
    except Exception as e:
        print(f"AI/JSON Error: {e}")
        # Clean up files on error
        for f in files_to_send:
            try:
                genai.delete_file(f.name)
            except:
                pass
        return json.dumps({"visual_fields": [], "confirmation_message": "Error processing form logic."})

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

    audio_file = None
    max_retries = 3
    retry_delay = 1
    
    try:
        # Upload audio file with timeout protection (optimized)
        audio_file = genai.upload_file(path=audio_path)
        upload_timeout = 25  # Reduced timeout
        upload_start = time.time()
        
        while audio_file.state.name == "PROCESSING":
            if time.time() - upload_start > upload_timeout:
                raise TimeoutError("Audio upload timeout")
            time.sleep(0.3)  # Faster checking interval (reduced from 0.5s)
            audio_file = genai.get_file(audio_file.name)

        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            system_instruction=SYSTEM_INSTRUCTION
        )
        
        # Optimized prompt - shorter for faster processing
        prompt = f"Analyze audio. Context: {user_prompt}. Language: {target_lang}. Output JSON only, plain text values."

        # Retry logic with exponential backoff
        response = None
        for attempt in range(max_retries):
            try:
                # Set explicit timeout (45 seconds per attempt - reduced for faster failure)
                response = model.generate_content(
                    [audio_file, prompt],
                    generation_config={
                        "response_mime_type": "application/json",
                        "temperature": 0.2,
                        "max_output_tokens": 800  # Further reduced for faster response
                    },
                    request_options={"timeout": 45}  # Reduced timeout for faster retries
                )
                
                # Verify response is valid
                if response and hasattr(response, 'text') and response.text:
                    # Success - break out of retry loop
                    break
                else:
                    raise ValueError("Invalid response from API")
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Check if it's a timeout/deadline error
                if "deadline" in error_str or "timeout" in error_str or "504" in error_str:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                        print(f"⚠️ Timeout on attempt {attempt + 1}/{max_retries}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        # Last attempt failed - return error response
                        raise TimeoutError(f"Audio analysis timed out after {max_retries} attempts")
                else:
                    # Non-timeout error - re-raise immediately
                    raise
        
        # Verify we have a valid response
        if not response or not hasattr(response, 'text'):
            raise ValueError("No valid response received from API")
        
        # Clean up uploaded file
        if audio_file:
            try:
                genai.delete_file(audio_file.name)
            except:
                pass  # Ignore cleanup errors
        
        # Process response
        data = json.loads(response.text)
        
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

    except TimeoutError as e:
        print(f"⏱️ Timeout Error: {e}")
        return json.dumps({
            "valid_audio": True,
            "condition": "Processing Timeout",
            "disease_type": "System Timeout",
            "simple_explanation": "The analysis took too long. Please try with a shorter audio clip or check your connection.",
            "recommendation": "Try recording a shorter audio sample (under 30 seconds).",
            "acoustic_analysis": "N/A",
            "severity": "Unknown"
        })
    except Exception as e:
        print(f"❌ Logic Error: {e}")
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
            "simple_explanation": "Could not process audio data safely. Please try again.",
            "recommendation": "Check internet connection and try again.",
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
        result = json.loads(res_str)
        return jsonify(amazon_translate_dict(result, lang))
    except json.JSONDecodeError as e:
        print(f"❌ JSON Decode Error: {e}")
        return jsonify({
            "error": "Invalid response format",
            "valid_audio": True,
            "condition": "Processing Error",
            "simple_explanation": "Could not parse the analysis result. Please try again.",
            "recommendation": "Retry with a shorter audio clip."
        }), 500
    except Exception as e:
        print(f"❌ Analyze Route Error: {e}")
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "valid_audio": True,
            "condition": "System Error",
            "simple_explanation": "An error occurred during analysis. Please try again.",
            "recommendation": "Check your connection and retry."
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