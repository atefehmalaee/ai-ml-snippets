"""
Save/load models (and pipelines) with joblib.
"""
from joblib import dump, load
from pathlib import Path
from datetime import datetime

def save_model(model, path: str, use_timestamp=True):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if use_timestamp:
        stem = Path(path).stem
        suffix = Path(path).suffix or ".joblib"
        path = str(Path(path).with_name(f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{suffix}"))
    dump(model, path)
    print(f"✅ Saved model to {path}")
    return path

def load_model(path: str):
    m = load(path)
    print(f"✅ Loaded model from {path}")
    return m
