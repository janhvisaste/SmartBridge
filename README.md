
# Harmonizing Health and Machine — Mastitis Detection (Farmer-Minimal, PDF-Enabled)

This project implements an AI-driven mastitis early detection system for dairy cows, 
designed to be farmer-friendly and require only simple field observations.

## Problem Statement
On smart farms, sensors and manual observations are collected from cows. A machine learning 
model analyzes these inputs to detect early signs of mastitis, alerting the farmer before 
symptoms become severe — reducing treatment costs and improving animal welfare.

## Key Features
- **Farmer-Minimal Inputs Only:**  
  - Body temperature (°C)  
  - Milk appearance (Normal / Abnormal)  
  - Udder hardness (0–2 scale)  
  - Udder pain (0–2 scale)  
- **Human-Friendly UI (Streamlit):**  
  - Single, simple screen.  
  - Clear field labels and tooltips.  
  - Risk level displayed with a **colored banner** (LOW, MEDIUM, HIGH, URGENT).  
- **Interpretation Panel:**  
  - Bullet-point summary of cow condition.  
  - Narrative explanation with suggested farmer/vet actions.  
- **Downloadable PDF Report:**  
  - After prediction, generate a printable PDF summary for farm/vet records.  
- **Batch Inference:** Upload a CSV file for multiple cows at once.  
- **Modular ML Pipeline:** Extendable to new diseases/species.  

## Project Structure
```
requirements.txt              # Python dependencies
config/config.yaml            # Config: target, model, thresholds, drop_columns
src/
 ├─ utils.py                  # YAML/JSON helpers
 ├─ data_loaders.py           # Data loading/cleaning
 ├─ preprocess.py             # Feature preprocessing (numeric + categorical)
 ├─ train.py                  # Training pipeline (LogReg by default)
 └─ risk.py                   # Risk thresholds and mapping
app/
 └─ streamlit_app.py          # Streamlit UI app
data/
 └─ clinical_mastitis_cows_version1.csv  # Example dataset
models/                       # Model artifacts after training
 ├─ best_model.joblib
 ├─ best_model_name.txt
 ├─ schema.json
 └─ leaderboard.json
```

## Installation
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate    # (Windows: .venv\Scripts\activate)

# Install requirements
pip install -r requirements.txt
```

## Usage

### 1. Train the model
Train on the provided dataset (ignores advanced sensor fields by config):
```bash
python -m src.train --csv data/clinical_mastitis_cows_version1.csv --out_dir models
```

Artifacts generated in `models/`:  
- `best_model.joblib` → trained classifier  
- `schema.json` → feature schema  
- `leaderboard.json` → performance metrics  

### 2. Launch the app
```bash
streamlit run app/streamlit_app.py
```

### 3. In the UI
- Enter **Temperature, Milk appearance, Hardness, Pain**.  
- Click **Predict**.  
- See risk probability, colored risk banner, situation summary, and interpretation.  
- Click **Download PDF summary** to save a report.  
- Use **Batch (CSV)** tab to predict for multiple cows.  

## Model
- Default: Logistic Regression (`class_weight=balanced`, `max_iter=300`).  
- Configurable in `config/config.yaml` (add RandomForest, XGBoost, LightGBM).  
- Risk thresholds (`urgent/high/low`) are defined in config.  

## PDF Report
- One-page summary with inputs, probability, risk, and interpretation.  
- Generated with ReportLab, saved directly from the UI.  
- Useful for farm logs and vet documentation.  

## Business/Social Impact
- **Early detection** → prevents severe mastitis, lowers treatment cost.  
- **Farmer-friendly** → requires no advanced sensors.  
- **Sustainability** → reduces milk loss and antibiotic usage.  
- **Record-keeping** → PDF reports can be archived or shared with vets.  

## Extensibility
- Add new input features (e.g., milk yield, rumination, barn sensors).  
- Extend to other livestock diseases (same pipeline).  
- Integrate IoT data ingestion (MQTT, Kafka).  
- Add alerting (SMS/WhatsApp) for urgent risk cases.  

---
© 2025 Harmonizing Health and Machine
