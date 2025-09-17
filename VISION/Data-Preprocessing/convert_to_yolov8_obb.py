#!/usr/bin/env python3
"""
Convert CVAT XML annotations to YOLOv8-OBB format
This script converts oriented bounding box annotations from CVAT XML format
to YOLOv8-OBB training format with train/val split and shuffling.
"""

import xml.etree.ElementTree as ET
import os
import shutil
import random
import math
import numpy as np
from pathlib import Path
import yaml
from typing import List, Tuple, Dict

def parse_cvat_xml(xml_path: str) -> Dict:
    """
    Parse CVAT XML file and extract annotations
    
    Args:
        xml_path: Path to the CVAT XML file
        
    Returns:
        Dictionary containing parsed annotations
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Extract image dimensions
    width = int(root.find('.//width').text)
    height = int(root.find('.//height').text)
    
    # Extract labels
    labels = {}
    for label in root.findall('.//label'):
        name = label.find('name').text
        labels[name] = len(labels)  # Assign class index
    
    # Extract annotations
    annotations = {}
    
    for track in root.findall('.//track'):
        label = track.get('label')
        class_id = labels[label]
        
        for box in track.findall('box'):
            frame = int(box.get('frame'))
            outside = int(box.get('outside'))
            
            # Skip boxes marked as outside
            if outside == 1:
                continue
                
            # Extract bounding box coordinates
            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))
            rotation = float(box.get('rotation', 0))
            
            # Convert to YOLOv8-OBB format
            # YOLOv8-OBB expects: class_id x1 y1 x2 y2 x3 y3 x4 y4
            # where coordinates are normalized (0-1) representing 4 corners of rotated rectangle
            
            x_center = (xtl + xbr) / 2.0 / width
            y_center = (ytl + ybr) / 2.0 / height
            bbox_width = (xbr - xtl) / width
            bbox_height = (ybr - ytl) / height
            
            # Normalize rotation to [-90, 90] degrees
            rotation_deg = rotation % 180
            if rotation_deg > 90:
                rotation_deg -= 180
            
            # Convert to 8 corner coordinates
            corner_coords = convert_bbox_to_obb_corners(
                x_center, y_center, bbox_width, bbox_height, rotation_deg
            )
            
            if frame not in annotations:
                annotations[frame] = []
            
            # Format: class_id x1 y1 x2 y2 x3 y3 x4 y4
            annotation = [class_id] + corner_coords
            annotations[frame].append(annotation)
    
    return {
        'annotations': annotations,
        'labels': labels,
        'image_size': (width, height)
    }

def sort_corners_clockwise(corners: List[Tuple[float, float]]) -> List[float]:
    """
    Sort corners in clockwise order starting from the one with the highest angle.
    
    Args:
        corners: List of (x, y) tuples
        
    Returns:
        Flattened list of sorted coordinates
    """
    # Calculate center
    center_x = sum(x for x, _ in corners) / 4
    center_y = sum(y for _, y in corners) / 4
    
    # Function to get angle from center (atan2 returns -pi to pi)
    def angle_from_center(point: Tuple[float, float]) -> float:
        return math.atan2(point[1] - center_y, point[0] - center_x)
    
    # Sort by angle in descending order for clockwise
    sorted_corners = sorted(corners, key=angle_from_center, reverse=True)
    
    # Flatten back to list
    return [coord for point in sorted_corners for coord in point]

def convert_bbox_to_obb_corners(x_center: float, y_center: float, 
                               width: float, height: float, 
                               rotation: float) -> List[float]:
    """
    Convert bounding box to YOLOv8-OBB format (8 corner coordinates)
    
    Args:
        x_center, y_center: Normalized center coordinates
        width, height: Normalized dimensions
        rotation: Rotation in degrees
        
    Returns:
        List of 8 coordinates: [x1, y1, x2, y2, x3, y3, x4, y4]
    """
    # Convert to radians
    angle_rad = math.radians(rotation)
    
    # Calculate half dimensions
    half_w = width / 2.0
    half_h = height / 2.0
    
    # Calculate corner points relative to center
    relative_corners = [
        (-half_w, -half_h),  # Top-left
        (half_w, -half_h),   # Top-right
        (half_w, half_h),    # Bottom-right
        (-half_w, half_h)    # Bottom-left
    ]
    
    # Rotate corners
    rotated_relative = []
    for x, y in relative_corners:
        new_x = x * math.cos(angle_rad) - y * math.sin(angle_rad)
        new_y = x * math.sin(angle_rad) + y * math.cos(angle_rad)
        rotated_relative.append((new_x, new_y))
    
    # Translate to center position
    translated_corners = [(x_center + x, y_center + y) for x, y in rotated_relative]
    
    # Sort in clockwise order
    sorted_coords = sort_corners_clockwise(translated_corners)
    
    # Clip to [0,1] range to prevent invalid coordinates
    for i in range(8):
        sorted_coords[i] = max(0.0, min(1.0, sorted_coords[i]))
    
    return sorted_coords

def create_yolo_dataset(annotations_data: Dict, images_dir: str, 
                       output_dir: str, train_ratio: float = 0.8, 
                       random_seed: int = 42) -> None:
    """
    Create YOLOv8-OBB dataset with train/val split
    
    Args:
        annotations_data: Parsed annotations data
        images_dir: Directory containing images
        output_dir: Output directory for YOLOv8 dataset
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
    
    # Get all image files from the images directory
    image_files = []
    for filename in os.listdir(images_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Extract frame number from filename (e.g., frame_000000.PNG -> 0)
            if filename.startswith('frame_') and filename.endswith('.PNG'):
                try:
                    frame_num = int(filename[6:12])  # Extract 6 digits after 'frame_'
                    image_files.append((frame_num, filename))
                except ValueError:
                    continue
    
    # Sort by frame number
    image_files.sort(key=lambda x: x[0])
    
    # Ensure we have exactly 2024 images (frame_000000 to frame_002023)
    expected_frames = set(range(2024))  # 0 to 2023
    found_frames = set(frame_num for frame_num, _ in image_files)
    missing_frames = expected_frames - found_frames
    
    if missing_frames:
        print(f"Warning: Missing {len(missing_frames)} images from expected 2024 frames")
        print(f"Missing frame numbers: {sorted(list(missing_frames))[:10]}...")  # Show first 10
    
    print(f"Total images found: {len(image_files)}")
    print(f"Images with annotations: {len(annotations_data['annotations'])}")
    print(f"Images without annotations: {len(image_files) - len(annotations_data['annotations'])}")
    
    # Shuffle all images
    random.shuffle(image_files)
    
    # Split into train and validation
    split_idx = int(len(image_files) * train_ratio)
    train_images = image_files[:split_idx]
    val_images = image_files[split_idx:]
    
    print(f"Training images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")
    
    # Process training data
    for frame_num, filename in train_images:
        process_frame(frame_num, filename, annotations_data, images_dir, 
                     train_images_dir, train_labels_dir)
    
    # Process validation data
    for frame_num, filename in val_images:
        process_frame(frame_num, filename, annotations_data, images_dir, 
                     val_images_dir, val_labels_dir)
    
    # Create dataset YAML file
    create_dataset_yaml(output_dir, annotations_data['labels'])

def process_frame(frame_num: int, filename: str, annotations_data: Dict, 
                 images_dir: str, output_images_dir: str, 
                 output_labels_dir: str) -> None:
    """
    Process a single frame: copy image and create label file
    
    Args:
        frame_num: Frame number
        filename: Original filename
        annotations_data: Parsed annotations data
        images_dir: Source images directory
        output_images_dir: Output images directory
        output_labels_dir: Output labels directory
    """
    # Generate label filename
    label_filename = f"frame_{frame_num:06d}.txt"
    
    # Copy image
    src_image_path = os.path.join(images_dir, filename)
    dst_image_path = os.path.join(output_images_dir, filename)
    
    if os.path.exists(src_image_path):
        shutil.copy2(src_image_path, dst_image_path)
        
        # Create label file (even if no annotations)
        label_path = os.path.join(output_labels_dir, label_filename)
        
        with open(label_path, 'w') as f:
            if frame_num in annotations_data['annotations']:
                for annotation in annotations_data['annotations'][frame_num]:
                    # Format: class_id x1 y1 x2 y2 x3 y3 x4 y4
                    line = f"{annotation[0]} {annotation[1]:.6f} {annotation[2]:.6f} {annotation[3]:.6f} {annotation[4]:.6f} {annotation[5]:.6f} {annotation[6]:.6f} {annotation[7]:.6f} {annotation[8]:.6f}\n"
                    f.write(line)
            # If no annotations, create empty file (YOLOv8 expects label file for each image)
    else:
        print(f"Warning: Image {filename} not found")

def create_dataset_yaml(output_dir: str, labels: Dict) -> None:
    """
    Create dataset.yaml file for YOLOv8 training
    
    Args:
        output_dir: Output directory
        labels: Dictionary mapping label names to class IDs
    """
    yaml_content = {
        'path': os.path.abspath(output_dir),
        'train': 'train/images',
        'val': 'val/images',
        'nc': len(labels),
        'names': list(labels.keys())
    }
    
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"Dataset YAML created: {yaml_path}")

def main():
    """Main function to convert CVAT XML to YOLOv8-OBB format"""
    
    # Configuration
    xml_path = "annotations.xml"
    images_dir = "images"
    output_dir = "yolov8_obb_dataset"
    train_ratio = 0.8  # 80% for training, 20% for validation
    random_seed = 42
    
    print("Converting CVAT XML annotations to YOLOv8-OBB format...")
    print(f"XML file: {xml_path}")
    print(f"Images directory: {images_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Train ratio: {train_ratio}")
    
    # Check if input files exist
    if not os.path.exists(xml_path):
        print(f"Error: XML file {xml_path} not found!")
        return
    
    if not os.path.exists(images_dir):
        print(f"Error: Images directory {images_dir} not found!")
        return
    
    # Parse XML annotations
    print("\nParsing XML annotations...")
    annotations_data = parse_cvat_xml(xml_path)
    
    print(f"Found {len(annotations_data['labels'])} classes: {list(annotations_data['labels'].keys())}")
    print(f"Image size: {annotations_data['image_size']}")
    print(f"Total annotated frames: {len(annotations_data['annotations'])}")
    
    # Create YOLOv8 dataset
    print("\nCreating YOLOv8-OBB dataset...")
    create_yolo_dataset(annotations_data, images_dir, output_dir, train_ratio, random_seed)
    
    print(f"\nDataset conversion completed!")
    print(f"Output directory: {output_dir}")
    print(f"Dataset structure:")
    print(f"  {output_dir}/")
    print(f"  ├── train/")
    print(f"  │   ├── images/")
    print(f"  │   └── labels/")
    print(f"  ├── val/")
    print(f"  │   ├── images/")
    print(f"  │   └── labels/")
    print(f"  └── dataset.yaml")
    
    print(f"\nTo train YOLOv8-OBB model, use:")
    print(f"yolo obb train data={os.path.abspath(output_dir)}/dataset.yaml model=yolov8n-obb.pt epochs=100")

if __name__ == "__main__":
    main()
