# YOLOv11 Dataset Preparation Scripts

A collection of Python scripts to prepare, clean, split, and analyze datasets for YOLOv11 training. These scripts help in converting raw data into a format suitable for training with Ultralytics YOLO models.

## Scripts

### 1. `02-1-reviewer.py`
- **Purpose**: Checks for mismatched images and labels (images without labels, labels without images, empty labels).
- **Usage**: Place your `images` and `labels` directories in the same folder as this script and run it.

### 2. `02-clean.py`
- **Purpose**: Removes empty pairs and renames files sequentially (e.g., `frame_000000.jpg`).
- **Usage**: Place your `images` and `labels` directories in the same folder as this script and run it.

### 3. `04-createdataset.py`
- **Purpose**: Splits a merged dataset into train/val sets (80/20 split), shuffles, and renames files.
- **Usage**: Requires a `merged` directory containing `images` and `labels` subdirectories. Run the script to generate `train` and `val` folders.

### 4. `02-1-change class-id.py`
- **Purpose**: Updates class IDs in label files (specifically changes class ID 1 to 5).
- **Usage**: Place your `labels` directory in the same folder as this script and run it.

### 5. `dataset_stats.py` / `05-dataset_stats.py`
- **Purpose**: Generates dataset statistics (train/val split, annotation counts) and creates a `dataset.yaml` file for training.
- **Usage**: Run the script pointing to your dataset directory (default is `dataset`).

## Requirements

- Python 3.x
- `ultralytics` (for training, optional for these scripts)

## Installation

```bash
pip install -r requirements.txt
```
