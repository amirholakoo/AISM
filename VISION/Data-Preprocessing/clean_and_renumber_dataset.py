#!/usr/bin/env python3
"""
Clean dataset by removing images without annotations and renumbering files
This script removes empty label files and renumbers all files starting from 0
"""

import os
import shutil
import random
import yaml
from pathlib import Path
from typing import List, Tuple

def clean_and_renumber_dataset(input_dir: str, output_dir: str, 
                              train_ratio: float = 0.8, random_seed: int = 42):
    """
    Clean dataset by removing images without annotations and renumbering files
    
    Args:
        input_dir: Input dataset directory (yolov8_obb_dataset)
        output_dir: Output directory for cleaned dataset
        train_ratio: Ratio of data for training (default: 0.8)
        random_seed: Random seed for reproducibility
    """
    random.seed(random_seed)
    
    # Create output directories
    train_images_dir = os.path.join(output_dir, 'train', 'images')
    train_labels_dir = os.path.join(output_dir, 'train', 'labels')
    val_images_dir = os.path.join(output_dir, 'val', 'images')
    val_labels_dir = os.path.join(output_dir, 'val', 'labels')
    
    for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Get all images with annotations
    train_input_dir = os.path.join(input_dir, 'train')
    val_input_dir = os.path.join(input_dir, 'val')
    
    # Collect all images with non-empty labels
    images_with_annotations = []
    
    # Process training images
    train_images_path = os.path.join(train_input_dir, 'images')
    train_labels_path = os.path.join(train_input_dir, 'labels')
    
    if os.path.exists(train_images_path) and os.path.exists(train_labels_path):
        for filename in os.listdir(train_images_path):
            if filename.endswith('.PNG'):
                # Get corresponding label file
                label_filename = filename.replace('.PNG', '.txt')
                label_path = os.path.join(train_labels_path, label_filename)
                
                if os.path.exists(label_path):
                    # Check if label file has content
                    with open(label_path, 'r') as f:
                        content = f.read().strip()
                        if content:  # Non-empty label file
                            images_with_annotations.append(('train', filename, label_filename))
    
    # Process validation images
    val_images_path = os.path.join(val_input_dir, 'images')
    val_labels_path = os.path.join(val_input_dir, 'labels')
    
    if os.path.exists(val_images_path) and os.path.exists(val_labels_path):
        for filename in os.listdir(val_images_path):
            if filename.endswith('.PNG'):
                # Get corresponding label file
                label_filename = filename.replace('.PNG', '.txt')
                label_path = os.path.join(val_labels_path, label_filename)
                
                if os.path.exists(label_path):
                    # Check if label file has content
                    with open(label_path, 'r') as f:
                        content = f.read().strip()
                        if content:  # Non-empty label file
                            images_with_annotations.append(('val', filename, label_filename))
    
    print(f"Found {len(images_with_annotations)} images with annotations")
    
    # Shuffle all images
    random.shuffle(images_with_annotations)
    
    # Split into train and validation
    split_idx = int(len(images_with_annotations) * train_ratio)
    train_images = images_with_annotations[:split_idx]
    val_images = images_with_annotations[split_idx:]
    
    print(f"Training images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")
    
    # Copy and renumber training images
    for idx, (original_split, image_filename, label_filename) in enumerate(train_images):
        new_image_filename = f"image_{idx:06d}.PNG"
        new_label_filename = f"image_{idx:06d}.txt"
        
        # Copy image
        src_image_path = os.path.join(input_dir, original_split, 'images', image_filename)
        dst_image_path = os.path.join(train_images_dir, new_image_filename)
        shutil.copy2(src_image_path, dst_image_path)
        
        # Copy label
        src_label_path = os.path.join(input_dir, original_split, 'labels', label_filename)
        dst_label_path = os.path.join(train_labels_dir, new_label_filename)
        shutil.copy2(src_label_path, dst_label_path)
    
    # Copy and renumber validation images
    for idx, (original_split, image_filename, label_filename) in enumerate(val_images):
        new_image_filename = f"image_{idx:06d}.PNG"
        new_label_filename = f"image_{idx:06d}.txt"
        
        # Copy image
        src_image_path = os.path.join(input_dir, original_split, 'images', image_filename)
        dst_image_path = os.path.join(val_images_dir, new_image_filename)
        shutil.copy2(src_image_path, dst_image_path)
        
        # Copy label
        src_label_path = os.path.join(input_dir, original_split, 'labels', label_filename)
        dst_label_path = os.path.join(val_labels_dir, new_label_filename)
        shutil.copy2(src_label_path, dst_label_path)
    
    # Create dataset YAML file
    create_cleaned_dataset_yaml(output_dir, len(train_images), len(val_images))
    
    print(f"\nCleaned dataset created successfully!")
    print(f"Output directory: {output_dir}")
    print(f"Training images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")

def create_cleaned_dataset_yaml(output_dir: str, train_count: int, val_count: int):
    """
    Create dataset.yaml file for cleaned dataset
    
    Args:
        output_dir: Output directory
        train_count: Number of training images
        val_count: Number of validation images
    """
    yaml_content = {
        'path': os.path.abspath(output_dir),
        'train': 'train/images',
        'val': 'val/images',
        'nc': 1,  # Number of classes
        'names': ['sack']  # Class names
    }
    
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"Dataset YAML created: {yaml_path}")

def get_cleaned_dataset_stats(dataset_path: str):
    """Get statistics about the cleaned dataset"""
    
    train_images = []
    train_labels = []
    val_images = []
    val_labels = []
    
    # Count training files
    train_images_dir = os.path.join(dataset_path, 'train', 'images')
    train_labels_dir = os.path.join(dataset_path, 'train', 'labels')
    
    if os.path.exists(train_images_dir):
        train_images = [f for f in os.listdir(train_images_dir) if f.endswith('.PNG')]
    if os.path.exists(train_labels_dir):
        train_labels = [f for f in os.listdir(train_labels_dir) if f.endswith('.txt')]
    
    # Count validation files
    val_images_dir = os.path.join(dataset_path, 'val', 'images')
    val_labels_dir = os.path.join(dataset_path, 'val', 'labels')
    
    if os.path.exists(val_images_dir):
        val_images = [f for f in os.listdir(val_images_dir) if f.endswith('.PNG')]
    if os.path.exists(val_labels_dir):
        val_labels = [f for f in os.listdir(val_labels_dir) if f.endswith('.txt')]
    
    # Count annotations
    train_annotations = 0
    val_annotations = 0
    
    for label_file in train_labels:
        label_path = os.path.join(train_labels_dir, label_file)
        with open(label_path, 'r') as f:
            lines = f.readlines()
            train_annotations += len([line for line in lines if line.strip()])
    
    for label_file in val_labels:
        label_path = os.path.join(val_labels_dir, label_file)
        with open(label_path, 'r') as f:
            lines = f.readlines()
            val_annotations += len([line for line in lines if line.strip()])
    
    total_images = len(train_images) + len(val_images)
    total_annotations = train_annotations + val_annotations
    
    print("=" * 60)
    print("🧹 CLEANED DATASET STATISTICS")
    print("=" * 60)
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
    print(f"  Avg Annotations per Image: {train_annotations/len(train_images):.2f}")
    print()
    print("✅ Validation Set:")
    print(f"  Images: {len(val_images)} ({len(val_images)/total_images*100:.1f}%)")
    print(f"  Annotations: {val_annotations} ({val_annotations/total_annotations*100:.1f}%)")
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

def main():
    """Main function to clean and renumber dataset"""
    
    # Configuration
    input_dir = "yolov8_obb_dataset"
    output_dir = "yolov8_obb_cleaned_dataset"
    train_ratio = 0.8  # 80% for training, 20% for validation
    random_seed = 42
    
    print("🧹 Cleaning and Renumbering YOLOv8-OBB Dataset")
    print("=" * 50)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Train ratio: {train_ratio}")
    print()
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} not found!")
        print("Please run the conversion script first.")
        return
    
    # Clean and renumber dataset
    clean_and_renumber_dataset(input_dir, output_dir, train_ratio, random_seed)
    
    # Show statistics
    print("\n" + "=" * 50)
    get_cleaned_dataset_stats(output_dir)

if __name__ == "__main__":
    main()
