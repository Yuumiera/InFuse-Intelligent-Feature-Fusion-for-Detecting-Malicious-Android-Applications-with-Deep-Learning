import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import joblib

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

# =====================================================
# 1. MODEL ARCHITECTURES
# =====================================================
class ManifestMLP(nn.Module):
    def __init__(self, input_dim, dropout=0.30):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Dropout(dropout),
            
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Dropout(dropout),
            
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(25, 1)
        )

    def forward(self, x):
        return self.net(x)

    def feature_extractor(self):
        return nn.Sequential(*list(self.net.children())[:-1])


class RuntimeMLP(nn.Module):
    def __init__(self, input_dim, dropout=0.30):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.LayerNorm(100),
            nn.Dropout(dropout),
            
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.LayerNorm(50),
            nn.Dropout(dropout),
            
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.LayerNorm(25),
            nn.Dropout(dropout),
            
            nn.Linear(25, 1)
        )

    def forward(self, x):
        return self.net(x)

    def feature_extractor(self):
        return nn.Sequential(*list(self.net.children())[:-1])


class PretrainedIntermediateFusionMLP(nn.Module):
    def __init__(self, manifest_branch, runtime_branch, dropout=0.20):
        super().__init__()
        self.manifest_branch = manifest_branch
        self.runtime_branch = runtime_branch

        self.fusion_head = nn.Sequential(
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.LayerNorm(25),
            nn.Dropout(dropout),
            nn.Linear(25, 1)
        )

    def forward(self, x_manifest, x_runtime):
        z_manifest = self.manifest_branch(x_manifest)
        z_runtime = self.runtime_branch(x_runtime)
        z_fused = torch.cat([z_manifest, z_runtime], dim=1)
        logits = self.fusion_head(z_fused)
        return logits


# =====================================================
# 2. GLOBAL LOADERS (Load models once on startup)
# =====================================================
manifest_model = None
manifest_preprocessor = None
manifest_threshold = 0.5

runtime_model = None
runtime_scaler = None
runtime_feature_columns = None
runtime_threshold = 0.5

fusion_model = None
fusion_threshold = 0.5

def init_models():
    global manifest_model, manifest_preprocessor, manifest_threshold
    global runtime_model, runtime_scaler, runtime_feature_columns, runtime_threshold
    global fusion_model, fusion_threshold
    
    # 2.1 Manifest Model
    try:
        manifest_pt = os.path.join(MODELS_DIR, 'manifest_mlp_100_50_25.pt')
        manifest_pkl = os.path.join(MODELS_DIR, 'manifest_mlp_preprocessor.pkl')
        if os.path.exists(manifest_pt) and os.path.exists(manifest_pkl):
            with open(manifest_pkl, 'rb') as f:
                manifest_preprocessor = pickle.load(f)
            
            ckpt = torch.load(manifest_pt, map_location=DEVICE, weights_only=False)
            manifest_model = ManifestMLP(input_dim=ckpt["input_dim"], dropout=ckpt.get("dropout", 0.3)).to(DEVICE)
            manifest_model.load_state_dict(ckpt["model_state_dict"])
            manifest_model.eval()
            manifest_threshold = 0.5 # Default, or can be added if you tuned it
    except Exception as e:
        print(f"Warning: Could not load Manifest Model: {e}")

    # 2.2 Runtime (Static) Model
    try:
        runtime_pt = os.path.join(MODELS_DIR, 'model2_runtime_mlp_100_50_25.pt')
        runtime_scaler_path = os.path.join(MODELS_DIR, 'model2_runtime_mlp_scaler.pkl')
        runtime_cols_path = os.path.join(MODELS_DIR, 'model2_runtime_mlp_feature_columns.pkl')
        
        if os.path.exists(runtime_pt) and os.path.exists(runtime_scaler_path):
            runtime_scaler = joblib.load(runtime_scaler_path)
            runtime_feature_columns = joblib.load(runtime_cols_path)
            
            ckpt = torch.load(runtime_pt, map_location=DEVICE, weights_only=False)
            runtime_model = RuntimeMLP(input_dim=ckpt["input_dim"], dropout=ckpt.get("dropout", 0.3)).to(DEVICE)
            runtime_model.load_state_dict(ckpt["model_state_dict"])
            runtime_model.eval()
            runtime_threshold = ckpt.get("best_threshold", 0.5)
    except Exception as e:
        print(f"Warning: Could not load Runtime Model: {e}")

    # 2.3 Fusion Model
    try:
        fusion_pt = os.path.join(MODELS_DIR, 'pretrained_intermediate_fusion_100_50_25.pt')
        if os.path.exists(fusion_pt) and manifest_model is not None and runtime_model is not None:
            ckpt = torch.load(fusion_pt, map_location=DEVICE, weights_only=False)
            # Recreate branches
            m_branch = ManifestMLP(input_dim=manifest_model.net[0].in_features).to(DEVICE).feature_extractor()
            r_branch = RuntimeMLP(input_dim=runtime_model.net[0].in_features).to(DEVICE).feature_extractor()
            
            fusion_model = PretrainedIntermediateFusionMLP(m_branch, r_branch).to(DEVICE)
            fusion_model.load_state_dict(ckpt["model_state_dict"])
            fusion_model.eval()
            fusion_threshold = ckpt.get("best_threshold", 0.5)
    except Exception as e:
        print(f"Warning: Could not load Fusion Model: {e}")

# Load at startup
init_models()

# =====================================================
# 3. PREPROCESSING HELPERS
# =====================================================
def prep_manifest_features(features: dict):
    """ Converts raw manifest dict into normalized numpy array suitable for model """
    if not manifest_preprocessor:
        raise Exception("Manifest preprocessor not loaded.")
        
    numeric_cols = manifest_preprocessor["numeric_cols"]
    vocabularies = manifest_preprocessor["vocabularies"]
    scaler = manifest_preprocessor["scaler"]
    
    # Num Features
    num_vals = []
    for col in numeric_cols:
        val = features.get(col, 0)
        if isinstance(val, bool) or str(val).lower() == 'true': val = 1
        elif str(val).lower() == 'false': val = 0
        try: val = float(val)
        except: val = 0.0
        num_vals.append(val)
        
    X_num = np.array([num_vals], dtype=np.float32)
    if len(numeric_cols) > 0:
        X_num = scaler.transform(X_num).astype(np.float32)
    
    # List Features
    list_matrices = []
    for col, vocab in vocabularies.items():
        matrix = np.zeros((1, len(vocab)), dtype=np.float32)
        items = features.get(col, [])
        if not isinstance(items, list):
            items = []
        for item in items:
            if item in vocab:
                matrix[0, vocab[item]] = 1.0
        list_matrices.append(matrix)
        
    X_list = np.hstack(list_matrices).astype(np.float32) if list_matrices else np.empty((1, 0), dtype=np.float32)
    X_manifest = np.hstack([X_num, X_list]).astype(np.float32)
    
    return torch.tensor(X_manifest).to(DEVICE)

def prep_runtime_features(features: dict):
    if not runtime_scaler or not runtime_feature_columns:
        raise Exception("Runtime scaler/columns not loaded.")
        
    vals = []
    for col in runtime_feature_columns:
        val = features.get(col, 0)
        if isinstance(val, bool) or str(val).lower() == 'true': val = 1
        elif str(val).lower() == 'false': val = 0
        try: val = float(val)
        except: val = 0.0
        vals.append(val)
        
    X = np.array([vals], dtype=np.float32)
    X = runtime_scaler.transform(X).astype(np.float32)
    return torch.tensor(X).to(DEVICE)

# =====================================================
# 4. INFERENCE PIPELINE
# =====================================================
def run_manifest_model(features: dict) -> dict:
    if manifest_model is None:
        return {"model": "Manifest Model", "result": "ERROR", "confidence": 0, "details": "Model Not Found"}
        
    with torch.no_grad():
        x = prep_manifest_features(features)
        logits = manifest_model(x)
        prob = torch.sigmoid(logits).item()
        
    is_malware = prob >= manifest_threshold
    
    return {
        "model": "Manifest Model",
        "result": "MALWARE" if is_malware else "BENIGN",
        "confidence": round(prob * 100 if is_malware else (1 - prob) * 100, 2),
        "details": f"Manifest probability: {prob:.4f} (Threshold: {manifest_threshold})"
    }

def run_static_model(features: dict) -> dict:
    if runtime_model is None:
        return {"model": "Static Model", "result": "ERROR", "confidence": 0, "details": "Model Not Found"}
        
    with torch.no_grad():
        x = prep_runtime_features(features)
        logits = runtime_model(x)
        prob = torch.sigmoid(logits).item()
        
    is_malware = prob >= runtime_threshold
    
    return {
        "model": "Static Model",
        "result": "MALWARE" if is_malware else "BENIGN",
        "confidence": round(prob * 100 if is_malware else (1 - prob) * 100, 2),
        "details": f"Runtime probability: {prob:.4f} (Threshold: {runtime_threshold})"
    }

def run_fusion_model(manifest_features: dict, static_features: dict) -> dict:
    if fusion_model is None:
        return {"model": "Fusion Model", "result": "ERROR", "confidence": 0, "details": "Model Not Found"}
        
    with torch.no_grad():
        x_m = prep_manifest_features(manifest_features)
        x_r = prep_runtime_features(static_features)
        logits = fusion_model(x_m, x_r)
        prob = torch.sigmoid(logits).item()
        
    is_malware = prob >= fusion_threshold
    
    return {
        "model": "Fusion Model",
        "result": "MALWARE" if is_malware else "BENIGN",
        "confidence": round(prob * 100 if is_malware else (1 - prob) * 100, 2),
        "details": f"Fusion probability: {prob:.4f} (Threshold: {fusion_threshold})"
    }
