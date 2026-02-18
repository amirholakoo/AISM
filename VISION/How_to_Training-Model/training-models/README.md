# Training Models

Python training scripts for YOLO model training and export (NCNN/ONNX) focused on Raspberry Pi 5 deployment.

## Files

- `train.py`
- `train-akhal.py`
- `train-with-auggmentation.py`
- `train-without-aggmentation.py`
- `train-without-auggmentation-small.py`
- `train-without-auggmentation-416-medium.py`

## Requirements

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Dataset Structure

Expected dataset path: `dataset/`

```text
dataset/
  train/
    images/
    labels/
  val/
    images/
    labels/
  dataset.yaml
```

## Run

Run any training script, for example:

```bash
python train-without-aggmentation.py
```

## Output

Trained weights and exported model files are saved in project output folders created by each script (for example `yolov11_pi5/` and `exported_pi5/`).
