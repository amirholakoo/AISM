#!/usr/bin/env python3
"""
Display dataset statistics for the converted YOLOv8-OBB dataset
"""

import os
import glob
from pathlib import Path

def get_dataset_stats(dataset_path: str):
    """Get statistics about the converted dataset"""
    
    train_images = glob.glob(os.path.join(dataset_path, "train", "images", "*.PNG"))
    train_labels = glob.glob(os.path.join(dataset_path, "train", "labels", "*.txt"))
    val_images = glob.glob(os.path.join(dataset_path, "val", "images", "*.PNG"))
    val_labels = glob.glob(os.path.join(dataset_path, "val", "labels", "*.txt"))
    
    # Count total annotations
    train_annotations = 0
    val_annotations = 0
    train_empty_files = 0
    val_empty_files = 0
    
    for label_file in train_labels:
        with open(label_file, 'r') as f:
            lines = f.readlines()
            annotation_count = len([line for line in lines if line.strip()])
            train_annotations += annotation_count
            if annotation_count == 0:
                train_empty_files += 1
    
    for label_file in val_labels:
        with open(label_file, 'r') as f:
            lines = f.readlines()
            annotation_count = len([line for line in lines if line.strip()])
            val_annotations += annotation_count
            if annotation_count == 0:
                val_empty_files += 1
    
    total_images = len(train_images) + len(val_images)
    total_annotations = train_annotations + val_annotations
    
    print("=" * 50)
    print("YOLOv8-OBB Dataset Statistics")
    print("=" * 50)
    print(f"Dataset Path: {os.path.abspath(dataset_path)}")
    print()
    print("📊 Overall Statistics:")
    print(f"  Total Images: {total_images}")
    print(f"  Total Annotations: {total_annotations}")
    print(f"  Average Annotations per Image: {total_annotations/total_images:.2f}")
    print()
    print("🚂 Training Set:")
    print(f"  Images: {len(train_images)} ({len(train_images)/total_images*100:.1f}%)")
    print(f"  Annotations: {train_annotations} ({train_annotations/total_annotations*100:.1f}%)")
    print(f"  Empty label files: {train_empty_files} ({train_empty_files/len(train_images)*100:.1f}%)")
    print(f"  Avg Annotations per Image: {train_annotations/len(train_images):.2f}")
    print()
    print("✅ Validation Set:")
    print(f"  Images: {len(val_images)} ({len(val_images)/total_images*100:.1f}%)")
    print(f"  Annotations: {val_annotations} ({val_annotations/total_annotations*100:.1f}%)")
    print(f"  Empty label files: {val_empty_files} ({val_empty_files/len(val_images)*100:.1f}%)")
    print(f"  Avg Annotations per Image: {val_annotations/len(val_images):.2f}")
    print()
    print("📁 File Structure:")
    print(f"  {dataset_path}/")
    print(f"  ├── train/")
    print(f"  │   ├── images/ ({len(train_images)} files)")
    print(f"  │   └── labels/ ({len(train_labels)} files)")
    print(f"  ├── val/")
    print(f"  │   ├── images/ ({len(val_images)} files)")
    print(f"  │   └── labels/ ({len(val_labels)} files)")
    print(f"  └── dataset.yaml")
    print()
    print("🎯 Ready for YOLOv8-OBB Training!")
    print(f"Command: yolo obb train data={os.path.abspath(dataset_path)}/dataset.yaml model=yolov8n-obb.pt epochs=100")

if __name__ == "__main__":
    dataset_path = "yolov8_obb_dataset"
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset directory '{dataset_path}' not found!")
        print("Please run the conversion script first.")
    else:
        get_dataset_stats(dataset_path)
