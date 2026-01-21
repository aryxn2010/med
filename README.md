# Sahayak.ai 🏥 | Advanced Medical Intelligence Platform

**Sahayak.ai** is a multi-modal AI medical assistant designed to triage patients, analyze medical imagery, and automate clinical documentation. Powered by Google Gemini 1.5 Pro, Firestore, and Flask, it bridges the gap between complex medical data and actionable insights.


## 🚀 Key Features

### 1. 🫁 Bio-Acoustic Triage
Analyzes respiratory sounds (coughing, breathing) to identify conditions like Croup, Asthma, or Pneumonia using Universal Vector Matching.
- **Inputs:** Live Microphone Recording or Audio File Upload (.wav, .mp3).
- **Outputs:** Condition name, severity risk, and simple explanations.

### 2. 👁️ Visual Health Scanner
A three-mode computer vision engine:
- **💊 Pharmacology:** Analyzes medicine labels/pills for interactions and safety.
- **🩹 Dermatology:** Scans wounds/skin conditions, assesses severity (Home Care vs. ER), and tracks healing.
- **🥗 Nutrition:** Analyzes food images for nutritional breakdown and dietary suitability (e.g., for Diabetics).

### 3. 📝 Smart Data Entry & Form Filling
Automates clinical paperwork using Voice-to-Text and Generative Typesetting.
- **Voice Dictation:** Speak patient details, and the AI structures the data.
- **AI Typesetter:** Upload a **blank image/PDF form**, speak the details, and the AI physically "types" the text onto the image in the correct boxes.

### 4. 🌍 Multi-Language Support
Full UI and Analysis translation support for **English**, **Hindi**, and **Kannada**.

---

## 🛠️ Installation & Setup (Local Run)

Follow these exact steps to run the application on your own computer.

### Step 1: Download & Organize Files
1. Download this repository (or clone it).
2. **Crucial:** Ensure **ALL** files (`app.py`, `service-account.json`, `.env`, the `templates` folder, and the `static` folder) are placed inside the **SAME folder**. Do not separate them into different sub-directories unless specified.

### Step 2: Install Dependencies
Open your terminal or command prompt in that folder and run:
```bash
pip install -r requirements.txt

create a .env file and enter your api keys

GEMINI_API_KEY=YOUR_KEY
GOOGLE_SAFE_BROWSING_KEY=YOUR_KEY
AWS_ACCESS_KEY_ID=YOUR_KEY
AWS_SECRET_ACCESS_KEY=YOUR_KEY





