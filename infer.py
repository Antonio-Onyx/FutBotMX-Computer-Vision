import os
from pathlib import Path

import torch

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path("Ultralytics").resolve()))

from ultralytics import YOLO

assert torch.cuda.is_available(), "CUDA no disponible"
print(torch.cuda.get_device_name(0))


VIDEO_SOURCE = Path(r"C:\Users\onyxg\Downloads\video_para_pruebas.mp4")
MODEL_PATH = "runs/train/robot_soccer_v1/weights/best.pt"
INFER_PROJECT = str(Path("runs/infer").resolve())


def main():
    if not VIDEO_SOURCE.exists():
        raise FileNotFoundError(f"No existe el video de prueba: {VIDEO_SOURCE}")

    model = YOLO(MODEL_PATH)
    model.predict(
        source=str(VIDEO_SOURCE),
        conf=0.4,
        iou=0.5,
        save=True,
        project=INFER_PROJECT,
        name="test_v1",
    )


if __name__ == "__main__":
    main()
