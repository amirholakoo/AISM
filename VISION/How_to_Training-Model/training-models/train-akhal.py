#!/usr/bin/env python3
"""
Train YOLOv11n model with custom dataset
Clean & compatible version for Ultralytics YOLOv11
"""

import os
import torch
from ultralytics import YOLO


def check_gpu():
    """Check GPU availability"""
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"CUDA Version: {torch.version.cuda if torch.version.cuda else 'Not Available'}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU Detected: {gpu_name}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        return True
    print("⚠️ No GPU detected → Training will use CPU (Slow)")
    print("💡 To enable GPU (CUDA 12.9 detected):")
    print("   pip uninstall torch torchvision torchaudio")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129")
    return False


def validate_dataset(path: str):
    """Validate YOLO dataset format"""
    print("🔍 Validating dataset structure...")

    yaml_file = os.path.join(path, "dataset.yaml")
    if not os.path.exists(yaml_file):
        print(f"❌ dataset.yaml NOT found at: {yaml_file}")
        return False

    required = [
        "train/images",
        "train/labels",
        "val/images",
        "val/labels",
    ]

    for sub in required:
        full = os.path.join(path, sub)
        if not os.path.exists(full):
            print(f"❌ Missing directory: {full}")
            return False

    print("✅ Dataset OK")
    return True


def train_yolo(dataset_path: str, epochs: int = 1, size="n"):
    """Train YOLOv11 model"""

    print("\n🚀 Starting YOLOv11 Training")
    gpu = check_gpu()

    if not gpu:
        print("❌ GPU is required but not available. Please check CUDA installation.")
        return

    if not validate_dataset(dataset_path):
        print("❌ Dataset validation failed.")
        return

    model_name = f"yolov11{size}.pt"
    print(f"\n📥 Loading model: {model_name}")

    try:
        model = YOLO(model_name)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    dataset_yaml = os.path.join(dataset_path, "dataset.yaml")

    # ----------------------------
    #   YOLOv11 Recommended Args
    # ----------------------------
    train_args = {
        "data": dataset_yaml,
        "epochs": epochs,
        "imgsz": 640,
        "batch": 16,
        "device": 0,
        "workers": 8,
        "project": "yolov11n_training",
        "name": "akhal_detection",
        "save": True,
        "save_period": 50,
        "cache": False,

        # Optimizer
        "lr0": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,

        # Augmentation (good defaults)
        "degrees": 10,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 2,
        "perspective": 0.0005,
        "flipud": 0.0,
        "fliplr": 0.5,

        "val": True,
        "plots": True,
        "verbose": True,
    }

    print("\n🎯 Training Config:")
    for k, v in train_args.items():
        print(f"  {k}: {v}")

    print("\n🚀 Training Started...")
    try:
        model.train(**train_args)
        print("\n🎉 Training completed!")

        best_model = "yolov11n_training/akhal_detection/weights/best.pt"
        if os.path.exists(best_model):
            print(f"🏆 Best model saved at: {best_model}")

    except Exception as e:
        print(f"❌ Training failed: {e}")


def main():
    dataset_path = "dataset"
    epochs = 50
    model_size = "n"

    print("\n==============================")
    print("      YOLOv11 Training")
    print("==============================")
    print(f"Dataset: {dataset_path}")
    print(f"Epochs: {epochs}")
    print(f"Model: YOLOv11{model_size}")

    

    if not os.path.isdir(dataset_path):
        print(f"❌ Dataset directory '{dataset_path}' not found!")
        return

    train_yolo(dataset_path, epochs, model_size)


if __name__ == "__main__":
    main()
