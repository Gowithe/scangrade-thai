# model_loader.py
import os
import threading
from ultralytics import YOLO

_lock = threading.Lock()
_models = {}

def get_model(model_path: str):
    """
    โหลด YOLO model แบบ singleton ต่อ process
    - ถ้าเรียกซ้ำด้วย model_path เดิม -> ได้ instance เดิมกลับ
    """
    model_path = model_path.strip()
    if not model_path:
        raise ValueError("MODEL_PATH is empty")

    # กัน path ผิดแบบเงียบ
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YOLO model not found: {model_path}")

    with _lock:
        if model_path not in _models:
            print(f"[MODEL] Loading YOLO once: {model_path}")
            _models[model_path] = YOLO(model_path)
        return _models[model_path]
