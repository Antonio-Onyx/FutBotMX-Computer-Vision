from collections import Counter
from pathlib import Path
import random
import shutil

import torch

assert torch.cuda.is_available(), "CUDA no disponible"
print(torch.cuda.get_device_name(0))


CLASS_NAMES = ["robot_team_a", "robot_team_b", "ball"]
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RANDOM_SEED = 42
TRAIN_RATIO = 0.8


def collect_image_label_pairs(dataset_dir: Path):
    pairs = {}

    for split in ("train", "val"):
        image_dir = dataset_dir / "images" / split
        label_dir = dataset_dir / "labels" / split
        if not image_dir.exists():
            continue

        for image_path in image_dir.iterdir():
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                print(f"WARNING: sin label correspondiente, se omite: {image_path}")
                continue

            pairs[image_path.stem] = (image_path, label_path)

    return list(pairs.values())


def count_instances(label_paths):
    counts = Counter()

    for label_path in label_paths:
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if not parts:
                continue

            try:
                class_id = int(parts[0])
            except ValueError:
                print(f"WARNING: clase inválida en {label_path}: {line}")
                continue

            if 0 <= class_id < len(CLASS_NAMES):
                counts[CLASS_NAMES[class_id]] += 1
            else:
                print(f"WARNING: clase fuera de rango en {label_path}: {line}")

    return counts


def recreate_split_dirs(dataset_dir: Path):
    for relative_dir in ("images/train", "images/val", "labels/train", "labels/val"):
        split_dir = dataset_dir / relative_dir
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)


def copy_split(split_items, dataset_dir: Path, split_name: str):
    copied_labels = []

    for image_path, label_path in split_items:
        target_image = dataset_dir / "images" / split_name / image_path.name
        target_label = dataset_dir / "labels" / split_name / label_path.name
        shutil.copy2(image_path, target_image)
        shutil.copy2(label_path, target_label)
        copied_labels.append(target_label)

    return copied_labels


def write_data_yaml(dataset_dir: Path):
    absolute_dataset_path = dataset_dir.resolve().as_posix()
    names = "[" + ", ".join(f"'{name}'" for name in CLASS_NAMES) + "]"
    data_yaml = (
        f"path: {absolute_dataset_path}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {names}\n"
    )
    (dataset_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")


def main():
    dataset_dir = DATASET_DIR
    pairs = collect_image_label_pairs(dataset_dir)

    if not pairs:
        raise SystemExit("No se encontraron imágenes con labels válidos para dividir.")

    random.seed(RANDOM_SEED)
    random.shuffle(pairs)

    train_count = int(len(pairs) * TRAIN_RATIO)
    train_items = pairs[:train_count]
    val_items = pairs[train_count:]

    temp_dir = dataset_dir / "_split_tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    temp_pairs = []
    try:
        for image_path, label_path in pairs:
            temp_image = temp_dir / image_path.name
            temp_label = temp_dir / label_path.name
            shutil.copy2(image_path, temp_image)
            shutil.copy2(label_path, temp_label)
            temp_pairs.append((temp_image, temp_label))

        train_stems = {image_path.stem for image_path, _ in train_items}
        staged_train = [pair for pair in temp_pairs if pair[0].stem in train_stems]
        staged_val = [pair for pair in temp_pairs if pair[0].stem not in train_stems]

        recreate_split_dirs(dataset_dir)
        train_labels = copy_split(staged_train, dataset_dir, "train")
        val_labels = copy_split(staged_val, dataset_dir, "val")
        write_data_yaml(dataset_dir)

        all_counts = count_instances(train_labels + val_labels)
        print(f"Train: {len(staged_train)} imágenes")
        print(f"Val: {len(staged_val)} imágenes")
        print("Instancias por clase:")
        for class_name in CLASS_NAMES:
            print(f"  {class_name}: {all_counts[class_name]}")
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
