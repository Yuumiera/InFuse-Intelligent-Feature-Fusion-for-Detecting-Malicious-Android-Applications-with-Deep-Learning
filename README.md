<div align="center">
  <h1>🛡️ InFuse</h1>
  <p><strong>Intelligent Feature Fusion for Detecting Malicious Android Applications with Deep Learning</strong></p>
  
  <p>
    <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python Version" />
    <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg" alt="PyTorch" />
    <img src="https://img.shields.io/badge/FastAPI-Backend-009688.svg" alt="FastAPI" />
    <img src="https://img.shields.io/badge/Vanilla_JS-Frontend-F7DF1E.svg" alt="JavaScript" />
  </p>
</div>

<br/>

## 📌 Project Overview
**InFuse** is a cutting-edge web application designed to detect Android malware using Deep Learning. It analyzes `.apk` files by extracting static and manifest features, and evaluates them using highly trained Multi-Layer Perceptron (MLP) and Feature Fusion models.

With its cyberpunk-inspired, interactive frontend and a robust FastAPI backend powered by PyTorch, InFuse provides real-time malware analysis and confidence scoring.

## 🚀 Live Demo
- **Frontend (UI):** [Click Here to View Live (Vercel)](#) *(Not: Buraya kendi Vercel linkini eklemelisin)*
- **Backend API:** [Hosted on Render](https://infuse-intelligent-feature-fusion-for.onrender.com)

## 🧠 Deep Learning Models
The system utilizes three distinct AI pipelines for analysis:
1. **Manifest Model (MLP):** Analyzes AndroidManifest.xml features (permissions, intents, hardware components).
2. **Static/Runtime Model (MLP):** Analyzes deeper static features (API calls, opcodes, strings).
3. **Fusion Model (Intermediate Fusion):** Combines the hidden layers of both Manifest and Static models to provide a highly accurate final verdict.

## ⚙️ System Architecture
- **Frontend:** HTML5, CSS3 (Cyberpunk aesthetic, Glassmorphism, CSS Animations), Vanilla JavaScript (Drag & Drop functionality).
- **Backend:** Python, FastAPI, Uvicorn.
- **AI/ML Engine:** PyTorch, Scikit-learn, Androguard (for APK feature extraction).

## 🛠️ Local Installation & Setup

### Prerequisites
- Python 3.9+
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/Yuumiera/InFuse-Intelligent-Feature-Fusion-for-Detecting-Malicious-Android-Applications-with-Deep-Learning.git
cd InFuse-Intelligent-Feature-Fusion-for-Detecting-Malicious-Android-Applications-with-Deep-Learning
```

### 2. Backend Setup
Navigate to the backend directory and install dependencies:
```bash
cd backend
pip install -r requirements.txt
```

Start the FastAPI server:
```bash
uvicorn main:app --reload
```
The backend API will run on `http://localhost:8000`.

### 3. Frontend Setup
Simply open `frontend/index.html` in your browser. (The frontend fetches data from the backend. Make sure the API URL in `script.js` points to your active backend environment).

## 📡 API Reference
### Analyze APK
- **Endpoint:** `POST /analyze`
- **Content-Type:** `multipart/form-data`
- **Body Parameters:**
  - `file`: The `.apk` file to analyze.
  - `model_type`: Model selection (`manifest`, `static`, or `fusion`).
- **Response:**
  ```json
  {
    "model": "Fusion Model",
    "result": "MALWARE",
    "confidence": 98.45,
    "details": "Fusion probability: 0.9845 (Threshold: 0.5)"
  }
  ```

