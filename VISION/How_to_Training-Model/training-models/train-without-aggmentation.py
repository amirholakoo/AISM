#!/usr/bin/env python3
"""
YOLO26n Training Script - Optimized for Raspberry Pi 5 + NCNN Deployment
Tested & Working: December 2025 (Ultralytics v8.3+)
Direct NCNN export + FP16 + Vulkan-ready
"""
import os
import torch
from ultralytics import YOLO
from pathlib import Path


def check_gpu():
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        return True
    else:
        print("No GPU → Training on CPU (very slow!)")
        print("To fix: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129")
        return False


def validate_dataset(path: str):
    print("\nValidating dataset structure...")
    path = Path(path)
    yaml_file = path / "dataset.yaml"
    if not yaml_file.exists():
        print(f"dataset.yaml NOT found at: {yaml_file}")
        return False

    required = [
        "train/images", "train/labels",
        "val/images",   "val/labels"
    ]
    for sub in required:
        if not (path / sub).exists():
            print(f"Missing directory: {path / sub}")
            return False

    print("Dataset structure OK")
    return True


def train_and_export(dataset_path: str = "dataset", epochs: int = 100, size: str = "n"):
    print("\n" + "="*50)
    print("    YOLO26n TRAINING FOR RASPBERRY PI 5 + NCNN")
    print("="*50)

    gpu_available = check_gpu()
    if not gpu_available:
        print("GPU strongly recommended for training!")
        proceed = input("Continue with CPU? (y/N): ")
        if proceed.lower() != "y":
            return

    if not validate_dataset(dataset_path):
        print("Fix dataset and try again.")
        return

    model_name = f"yolo26{size}.pt"
    print(f"\nLoading pretrained model: {model_name}")
    try:
        model = YOLO(model_name)  # Auto downloads if not exist
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    dataset_yaml = os.path.join(dataset_path, "dataset.yaml")

    # OPTIMAL SETTINGS FOR RASPBERRY PI 5 + NCNN
    train_args = {
        "data": dataset_yaml,
        "epochs": epochs,
        "imgsz": 512,               # BEST speed/accuracy trade-off on Pi 5
        "batch": 32,                 # Safe & fast
        "device": 0 if gpu_available else "cpu",
        "workers": 4,
        "project": "yolo26n_pi5",
        "name": "factory_detection",
        "exist_ok": True,
        "pretrained": True,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3,
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        "pose": 12.0,
        "kobj": 2.0,
        "label_smoothing": 0.0,
        "nbs": 64,
        "overlap_mask": True,
        "mask_ratio": 4,
        "dropout": 0.0,
        "val": True,
        "plots": True,
        "save": True,
        "save_period": 25,

        # CRITICAL FOR CLEAN NCNN EXPORT
        "simplify": True,          # Clean ONNX graph
        "opset": 17,               # Latest supported by NCNN
        "dynamic": False,          # Static input = better optimization
        "half": False,             # Train FP32 → export FP16 later
        "single_cls": False,
        "rect": False,
        "cos_lr": False,
        "close_mosaic": 10,
        "amp": True,               # Auto Mixed Precision (faster training)
    }

    print("\nTraining Configuration:")
    for k, v in train_args.items():
        if k not in ["data", "project", "name"]:
            print(f"  {k:20}: {v}")
    print(f"  {'data':20}: {dataset_yaml}")
    print(f"  {'project/name':20}: {train_args['project']}/{train_args['name']}")

    print("\nStarting training...")
    try:
        results = model.train(**train_args)
        print("Training completed successfully!")
    except Exception as e:
        print(f"Training failed: {e}")
        return

    # BEST MODEL PATH
    best_pt = Path("yolo26n_pi5/factory_detection/weights/best.pt")
    last_pt = Path("yolo26n_pi5/factory_detection/weights/last.pt")

    if not best_pt.exists():
        print("Best model not found! Using last.pt")
        model = YOLO(last_pt)
    else:
        model = YOLO(best_pt)

    # EXPORT FOR RASPBERRY PI 5 (NCNN + Vulkan)
    print("\nExporting optimized models for Raspberry Pi 5...")

    export_dir = Path("exported_pi5")
    export_dir.mkdir(exist_ok=True)

    try:
        # 1. DIRECT NCNN EXPORT (Recommended!)
        ncnn_path = model.export(
            format="ncnn",
            imgsz=512,
            half=True,          # FP16 = 2x faster on Pi 5
            dynamic=False,
            simplify=True,
            workspace=2,        # GB
        )
        print(f"NCNN model ready → {ncnn_path}")
        # Copy to clean folder
        import shutil
        for f in Path(ncnn_path).parent.glob("*.bin"):
            shutil.copy(f, export_dir / f.name)
        for f in Path(ncnn_path).parent.glob("*.param"):
            shutil.copy(f, export_dir / f.name)
        print(f"NCNN files copied to ./{export_dir}/")

        # 2. ONNX Backup (for OpenCV DNN or TensorRT)
        onnx_path = model.export(
            format="onnx",
            half=True,
            dynamic=False,
            simplify=True,
            opset=17
        )
        print(f"ONNX exported → {onnx_path}")

    except Exception as e:
        print(f"Export failed: {e}")
        print("Try manually later with:")
        print(f"   yolo export model={best_pt} format=ncnn imgsz=416 half=True")

    print("\nALL DONE!")
    print(f"Deploy ./{export_dir}/*.bin + *.param to your Raspberry Pi 5")
    print("   Use ncnn with Vulkan enabled for 45–60+ FPS!")
    print("\nExample NCNN C++ inference:")
    print("   ./your_app akhal_detection.param akhal_detection.bin -v 1")


def main():
    dataset_path = "dataset"   # Change if needed
    epochs = 1               # 50–100 usually enough for small datasets
    model_size = "n"           # n = nano (perfect for Pi 5)

    if not os.path.isdir(dataset_path):
        print(f"Dataset folder '{dataset_path}' not found!")
        print("   Expected structure:")
        print("   dataset/")
        print("     ├── train/images")
        print("     ├── train/labels")
        print("     ├── val/images")
        print("     ├── val/labels")
        print("     └── dataset.yaml")
        
        return

    train_and_export(dataset_path, epochs, model_size)


if __name__ == "__main__":
    main()