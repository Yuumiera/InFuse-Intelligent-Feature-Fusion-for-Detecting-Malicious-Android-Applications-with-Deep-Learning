from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import os
import shutil

from feature_extractor import extract_all
from model_runner import run_manifest_model, run_static_model, run_fusion_model

app = FastAPI(title="InFuse Malware Analysis API")

# CORS izinleri
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = "temp_apks"
os.makedirs(TEMP_DIR, exist_ok=True)

# API Endpoint for Analysis
@app.post("/analyze")
async def analyze_apk(
    file: UploadFile = File(...),
    model_type: str = Form(...) # 'manifest', 'static', veya 'fusion'
):
    if not file.filename.endswith('.apk'):
        return JSONResponse(status_code=400, content={"error": "Sadece .apk dosyaları kabul edilir."})
    
    file_path = os.path.join(TEMP_DIR, file.filename)
    
    try:
        # Dosyayı kaydet
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Özellikleri çıkar
        # Gerçekte asenkron veya background task olarak yapılmalı, şimdilik senkron yapıyoruz.
        features = extract_all(file_path)
        
        # Seçilen modele göre tahmin yap
        if model_type == 'manifest':
            result = run_manifest_model(features['manifest'])
        elif model_type == 'static':
            result = run_static_model(features['static'])
        elif model_type == 'fusion':
            result = run_fusion_model(features['manifest'], features['static'])
        else:
            return JSONResponse(status_code=400, content={"error": "Geçersiz model_type."})
            
        return result
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
        
    finally:
        # İşlem bitince temp dosyasını temizle
        if os.path.exists(file_path):
            os.remove(file_path)

# Frontend klasörünü serve et
app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    # uvicorn main:app --reload
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
