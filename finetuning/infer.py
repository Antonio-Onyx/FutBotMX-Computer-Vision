import os
from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(BASE_DIR / "Ultralytics"))

from ultralytics import YOLO

assert torch.cuda.is_available(), "CUDA no disponible"
print(torch.cuda.get_device_name(0))


VIDEO_SOURCE = Path(r"C:\Users\onyxg\Downloads\video_para_pruebas.mp4")
MODEL_PATH = BASE_DIR / "models" / "YOLOv8n_robots.pt"
INFER_PROJECT = BASE_DIR / "runs" / "infer"


def main():
    if not VIDEO_SOURCE.exists():
        raise FileNotFoundError(f"No existe el video de prueba: {VIDEO_SOURCE}")

    model = YOLO(str(MODEL_PATH))
    model.predict(
        source=str(VIDEO_SOURCE),
        conf=0.4,
        iou=0.5,
        save=True,
        project=str(INFER_PROJECT),
        name="test_v1",
    )


if __name__ == "__main__":
    main()
