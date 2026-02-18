#!/usr/bin/env python3
"""
Display dataset statistics for the converted YOLOv11n dataset
"""

import os
import glob


def safe_ratio(numerator: float, denominator: float) -> float:
    """Return numerator/denominator, guarding against division by zero."""
    return numerator / denominator if denominator else 0.0


def summarize_label_files(label_files, class_ids):
    """Count annotations, empty files, and collect class ids from label files."""
    annotations = 0
    empty_files = 0
    for label_file in label_files:
        with open(label_file, 'r') as f:
            entries = [line.strip() for line in f if line.strip()]
        annotations += len(entries)
        if not entries:
            empty_files += 1
            continue
        for entry in entries:
            parts = entry.split()
            if not parts:
                continue
            try:
                class_ids.add(int(float(parts[0])))
            except ValueError:
                continue
    return annotations, empty_files


def discover_class_names(dataset: str):
    """Look for classes.txt/names.txt near the dataset."""
    candidate_dirs = [
        dataset,
        os.path.join(dataset, "train"),
        os.path.join(dataset, "train", "labels"),
        os.path.join(dataset, "val"),
        os.path.join(dataset, "val", "labels"),
    ]
    filenames = ("classes.txt", "names.txt")
    for directory in candidate_dirs:
        if not os.path.isdir(directory):
            continue
        for filename in filenames:
            candidate = os.path.join(directory, filename)
            if os.path.isfile(candidate):
                with open(candidate, "r", encoding="utf-8") as f:
                    names = [line.strip() for line in f if line.strip()]
                    if names:
                        return names
    return []


def ensure_dataset_yaml(dataset: str, class_ids):
    """Create dataset.yaml if it does not exist."""
    dataset_yaml = os.path.join(dataset, "dataset.yaml")
    if os.path.exists(dataset_yaml):
        return dataset_yaml, False

    names = discover_class_names(dataset)
    if not names:
        if class_ids:
            max_class = max(class_ids)
            names = [f"class_{idx}" for idx in range(max_class + 1)]
        else:
            names = ["object"]

    if class_ids:
        required_len = max(class_ids) + 1
        if len(names) < required_len:
            names.extend(f"class_{idx}" for idx in range(len(names), required_len))

    yaml_lines = [
        f"path: {os.path.abspath(dataset)}",
        "train: train/images",
        "val: val/images",
    ]

    if os.path.isdir(os.path.join(dataset, "test", "images")):
        yaml_lines.append("test: test/images")

    yaml_lines.append(f"nc: {len(names)}")
    yaml_lines.append("names:")
    for idx, name in enumerate(names):
        yaml_lines.append(f"  {idx}: {name}")

    with open(dataset_yaml, "w", encoding="utf-8") as f:
        f.write("\n".join(yaml_lines) + "\n")

    print(f"✅ Created dataset.yaml at: {dataset_yaml}")
    print("   Update the class names in dataset.yaml if needed.")
    return dataset_yaml, True

def get_dataset_stats(dataset: str):
    """Get statistics about the converted dataset"""
    
    train_images = glob.glob(os.path.join(dataset, "train", "images", "*.PNG"))
    train_labels = glob.glob(os.path.join(dataset, "train", "labels", "*.txt"))
    val_images = glob.glob(os.path.join(dataset, "val", "images", "*.PNG"))
    val_labels = glob.glob(os.path.join(dataset, "val", "labels", "*.txt"))
    
    # Count total annotations
    class_ids = set()
    train_annotations, train_empty_files = summarize_label_files(train_labels, class_ids)
    val_annotations, val_empty_files = summarize_label_files(val_labels, class_ids)
    
    total_images = len(train_images) + len(val_images)
    total_annotations = train_annotations + val_annotations
    dataset_yaml_path, yaml_created = ensure_dataset_yaml(dataset, class_ids)

    avg_annotations = safe_ratio(total_annotations, total_images)
    train_image_pct = safe_ratio(len(train_images), total_images) * 100
    train_annotation_pct = safe_ratio(train_annotations, total_annotations) * 100
    train_empty_pct = safe_ratio(train_empty_files, len(train_images)) * 100
    train_avg_annotations = safe_ratio(train_annotations, len(train_images))

    val_image_pct = safe_ratio(len(val_images), total_images) * 100
    val_annotation_pct = safe_ratio(val_annotations, total_annotations) * 100
    val_empty_pct = safe_ratio(val_empty_files, len(val_images)) * 100
    val_avg_annotations = safe_ratio(val_annotations, len(val_images))
    
    print("=" * 50)
    print("YOLOv11 Dataset Statistics")
    print("=" * 50)
    print(f"Dataset Path: {os.path.abspath(dataset)}")
    print()
    print("📊 Overall Statistics:")
    print(f"  Total Images: {total_images}")
    print(f"  Total Annotations: {total_annotations}")
    print(f"  Average Annotations per Image: {avg_annotations:.2f}")
    if total_images == 0:
        print("  ⚠️ No images found; percentages default to 0.0.")
    print()
    print("🚂 Training Set:")
    print(f"  Images: {len(train_images)} ({train_image_pct:.1f}%)")
    print(f"  Annotations: {train_annotations} ({train_annotation_pct:.1f}%)")
    print(f"  Empty label files: {train_empty_files} ({train_empty_pct:.1f}%)")
    print(f"  Avg Annotations per Image: {train_avg_annotations:.2f}")
    print()
    print("✅ Validation Set:")
    print(f"  Images: {len(val_images)} ({val_image_pct:.1f}%)")
    print(f"  Annotations: {val_annotations} ({val_annotation_pct:.1f}%)")
    print(f"  Empty label files: {val_empty_files} ({val_empty_pct:.1f}%)")
    print(f"  Avg Annotations per Image: {val_avg_annotations:.2f}")
    print()
    print("📁 File Structure:")
    print(f"  {os.path.abspath(dataset)}/")
    print(f"  ├── train/")
    print(f"  │   ├── images/ ({len(train_images)} files)")
    print(f"  │   └── labels/ ({len(train_labels)} files)")
    print(f"  ├── val/")
    print(f"  │   ├── images/ ({len(val_images)} files)")
    print(f"  │   └── labels/ ({len(val_labels)} files)")
    yaml_status = "dataset.yaml"
    if not os.path.exists(dataset_yaml_path):
        yaml_status += " (missing)"
    elif yaml_created:
        yaml_status += " (created now)"
    print(f"  └── {yaml_status}")
    print()
    print("🎯 Ready for YOLOv11n Training!")
    print(f"Command: yolov11n train data={os.path.abspath(dataset)}/dataset.yaml model=yolo11n.pt epochs=50")

if __name__ == "__main__":
    dataset_path = "dataset"
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset directory '{dataset_path}' not found!")
        print("Please run the conversion script first.")
    else:
        get_dataset_stats(dataset_path)
