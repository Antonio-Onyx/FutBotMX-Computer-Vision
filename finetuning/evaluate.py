import os
from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(BASE_DIR / "Ultralytics"))

from ultralytics import YOLO

assert torch.cuda.is_available(), "CUDA no disponible"
print(torch.cuda.get_device_name(0))

DATA_YAML = BASE_DIR / "dataset" / "data.yaml"
MODEL_PATH = BASE_DIR / "models" / "YOLOv8n_robots.pt"
CLASS_NAMES = ["robot_team_a", "robot_team_b", "ball"]


def format_metric(value):
    if value is None:
        return "N/A"
    return f"{float(value):.4f}"


def class_metrics(metrics, class_index):
    box_metrics = getattr(metrics, "box", None)
    if box_metrics is None:
        return None, None, None, None

    if hasattr(box_metrics, "class_result"):
        precision, recall, map50, map50_95 = box_metrics.class_result(class_index)
        return precision, recall, map50, map50_95

    precision = getattr(box_metrics, "p", [None] * len(CLASS_NAMES))[class_index]
    recall = getattr(box_metrics, "r", [None] * len(CLASS_NAMES))[class_index]
    map50 = getattr(box_metrics, "ap50", [None] * len(CLASS_NAMES))[class_index]
    map50_95 = getattr(box_metrics, "maps", [None] * len(CLASS_NAMES))[class_index]
    return precision, recall, map50, map50_95


def main():
    model = YOLO(str(MODEL_PATH))
    metrics = model.val(data=str(DATA_YAML), split="val")

    print("\nMétricas por clase:")
    for class_index, class_name in enumerate(CLASS_NAMES):
        precision, recall, map50, map50_95 = class_metrics(metrics, class_index)
        print(
            f"{class_name}: "
            f"precision={format_metric(precision)}, "
            f"recall={format_metric(recall)}, "
            f"mAP50={format_metric(map50)}, "
            f"mAP50-95={format_metric(map50_95)}"
        )


if __name__ == "__main__":
    main()
