#!/usr/bin/env python3
"""
Train YOLOv8-OBB model with custom dataset
This script trains a YOLOv8-OBB model using the prepared dataset
"""

import os
import yaml
from ultralytics import YOLO
import torch
from pathlib import Path

def check_gpu_availability():
    """Check if GPU is available for training"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU Available: {gpu_name}")
        print(f"📊 GPU Count: {gpu_count}")
        return True
    else:
        print("⚠️  GPU not available, using CPU (training will be slower)")
        return False

def validate_dataset(dataset_path: str):
    """Validate dataset structure and files"""
    print("🔍 Validating dataset...")
    
    # Check if dataset.yaml exists
    yaml_path = os.path.join(dataset_path, 'dataset.yaml')
    if not os.path.exists(yaml_path):
        print(f"❌ Error: dataset.yaml not found at {yaml_path}")
        return False
    
    # Check train and val directories
    train_images_dir = os.path.join(dataset_path, 'train', 'images')
    train_labels_dir = os.path.join(dataset_path, 'train', 'labels')
    val_images_dir = os.path.join(dataset_path, 'val', 'images')
    val_labels_dir = os.path.join(dataset_path, 'val', 'labels')
    
    required_dirs = [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"❌ Error: Directory not found: {dir_path}")
            return False
    
    # Count files
    train_images = [f for f in os.listdir(train_images_dir) if f.endswith(('.PNG', '.png', '.jpg', '.jpeg'))]
    train_labels = [f for f in os.listdir(train_labels_dir) if f.endswith('.txt')]
    val_images = [f for f in os.listdir(val_images_dir) if f.endswith(('.PNG', '.png', '.jpg', '.jpeg'))]
    val_labels = [f for f in os.listdir(val_labels_dir) if f.endswith('.txt')]
    
    print(f"📊 Dataset Statistics:")
    print(f"  Training images: {len(train_images)}")
    print(f"  Training labels: {len(train_labels)}")
    print(f"  Validation images: {len(val_images)}")
    print(f"  Validation labels: {len(val_labels)}")
    
    # Check if image and label counts match
    if len(train_images) != len(train_labels):
        print(f"⚠️  Warning: Training images ({len(train_images)}) and labels ({len(train_labels)}) count mismatch")
    
    if len(val_images) != len(val_labels):
        print(f"⚠️  Warning: Validation images ({len(val_images)}) and labels ({len(val_labels)}) count mismatch")
    
    print("✅ Dataset validation completed")
    return True

def train_yolov8_obb(dataset_path: str, epochs: int = 100, model_size: str = 'n'):
    """
    Train YOLOv8-OBB model
    
    Args:
        dataset_path: Path to dataset directory containing dataset.yaml
        epochs: Number of training epochs
        model_size: Model size ('n', 's', 'm', 'l', 'x')
    """
    
    print("🚀 Starting YOLOv8-OBB Training")
    print("=" * 50)
    
    # Check GPU availability
    gpu_available = check_gpu_availability()
    
    # Validate dataset
    if not validate_dataset(dataset_path):
        print("❌ Dataset validation failed!")
        return False
    
    # Load model
    model_name = f"yolov8{model_size}-obb.pt"
    print(f"📥 Loading model: {model_name}")
    
    try:
        model = YOLO(model_name)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # Prepare dataset path
    dataset_yaml_path = os.path.join(dataset_path, 'dataset.yaml')
    
    # Training parameters
    training_args = {
        'data': dataset_yaml_path,
        'epochs': epochs,
        'imgsz': 640,  # Image size
        'batch': 64 if gpu_available else 4,  # Reduced batch size for stability
        'device': 0 if gpu_available else 'cpu',  # Use GPU if available
        'workers': 8 if gpu_available else 4,  # Number of workers
        'project': 'yolov8_obb_training',  # Project name
        'name': 'sack_detection',  # Experiment name
        'save': True,  # Save checkpoints
        'save_period': 10,  # Save checkpoint every 10 epochs
        'cache': True,  # Cache images for faster training
        'patience': 0,  # Disable early stopping
        'lr0': 0.001,  # Lower initial learning rate for stability
        'lrf': 0.001,  # Lower final learning rate
        'momentum': 0.937,  # SGD momentum
        'weight_decay': 0.001,  # Increased weight decay to prevent overfitting
        'warmup_epochs': 3,  # Warmup epochs
        'warmup_momentum': 0.8,  # Warmup momentum
        'warmup_bias_lr': 0.1,  # Warmup bias learning rate
        'box': 13.0,  # Increased box loss gain for tighter bounding boxes
        'cls': 1.5,  # Increased classification loss gain to reduce false positives
        'dfl': 2.0,  # Increased DFL loss for better box regression
        'pose': 12.0,  # Pose loss gain
        'kobj': 2.0,  # Keypoint object loss gain
        'label_smoothing': 0.0,  # Disable label smoothing
        'nbs': 64,  # Nominal batch size
        'overlap_mask': True,  # Overlap mask
        'mask_ratio': 4,  # Mask ratio
        'dropout': 0.0,  # Disable dropout
        'val': True,  # Validate during training
        'plots': True,  # Generate training plots
        'verbose': True,  # Verbose output
        'degrees': 0.0,  # Disable rotation augmentation
        'translate': 0.0,  # Disable translation
        'scale': 0.0,  # Disable scaling
        'shear': 0.0,  # Disable shearing
        'perspective': 0.0,  # Disable perspective distortion
        'flipud': 0.0,  # Disable vertical flip
        'fliplr': 0.0,  # Disable horizontal flip
    }
    
    print(f"🎯 Training Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Dataset: {dataset_yaml_path}")
    print(f"  Epochs: {epochs}")
    print(f"  Image Size: {training_args['imgsz']}")
    print(f"  Batch Size: {training_args['batch']}")
    print(f"  Device: {'GPU' if gpu_available else 'CPU'}")
    print(f"  Workers: {training_args['workers']}")
    print(f"  Project: {training_args['project']}")
    print(f"  Experiment: {training_args['name']}")
    
    print("\n🚀 Starting training...")
    print("=" * 50)
    
    try:
        # Start training
        results = model.train(**training_args)
        
        print("\n✅ Training completed successfully!")
        print("=" * 50)
        
        # Print training results
        if hasattr(results, 'results_dict'):
            print("📊 Training Results:")
            for key, value in results.results_dict.items():
                print(f"  {key}: {value}")
        
        # Save final model
        best_model_path = os.path.join('yolov8_obb_training', 'sack_detection', 'weights', 'best.pt')
        if os.path.exists(best_model_path):
            print(f"🏆 Best model saved at: {best_model_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

def main():
    """Main function to train YOLOv8-OBB model"""
    
    # Configuration
    dataset_path = "yolov8_obb_cleaned_dataset"  # Use cleaned dataset
    epochs = 100 # Increased default epochs for more thorough training
    model_size = 'n'  # Options: 'n', 's', 'm', 'l', 'x'
    
    print("🎯 YOLOv8-OBB Model Training")
    print("=" * 50)
    print(f"Dataset: {dataset_path}")
    print(f"Epochs: {epochs}")
    print(f"Model: YOLOv8{model_size}-OBB")
    print()
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Dataset directory '{dataset_path}' not found!")
        print("Please run the dataset preparation scripts first.")
        return
    
    # Check if ultralytics is installed
    try:
        import ultralytics
        print(f"✅ Ultralytics version: {ultralytics.__version__}")
    except ImportError:
        print("❌ Error: Ultralytics not installed!")
        print("Please install it with: pip install ultralytics")
        return
    
    # Start training
    success = train_yolov8_obb(dataset_path, epochs, model_size)
    
    if success:
        print("\n🎉 Training completed successfully!")
        print("📁 Check the 'yolov8_obb_training' directory for results")
        print("🏆 Best model: yolov8_obb_training/sack_detection/weights/best.pt")
        print("📊 Training plots: yolov8_obb_training/sack_detection/")
    else:
        print("\n❌ Training failed!")

if __name__ == "__main__":
    main()
