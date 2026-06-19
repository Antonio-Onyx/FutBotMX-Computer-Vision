import os
from pathlib import Path

import torch

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path("Ultralytics").resolve()))

from ultralytics import YOLO

assert torch.cuda.is_available(), "CUDA no disponible"
print(torch.cuda.get_device_name(0))

DATA_YAML = "dataset/data.yaml"
TRAIN_PROJECT = str(Path("runs/train").resolve())


def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data=DATA_YAML,
        epochs=150,
        imgsz=640,
        batch=16,
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=5,
        freeze=10,
        augment=True,
        mosaic=1.0,
        flipud=0.3,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.3,
        device=0,
        project=TRAIN_PROJECT,
        name="robot_soccer_v1",
        exist_ok=True,
        patience=30,
        save_period=10,
        val=True,
        plots=True,
    )


if __name__ == "__main__":
    main()
