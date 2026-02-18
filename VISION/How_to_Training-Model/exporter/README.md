# YOLO PyTorch to ONNX Exporter

This project exports a trained YOLO model from PyTorch weights (`.pt`) to ONNX format.

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

- `exporter-pt-onnx.py` - Main export script
- `weights/best.pt` - Your trained model weights

## Usage

Run:

```bash
python exporter-pt-onnx.py
```

The script exports the model to ONNX using:
- `dynamic=True`
- `imgsz=640`
- `opset=12`
- `simplify=True`

## Notes

- Make sure `weights/best.pt` exists before running.
- Change export options in `exporter-pt-onnx.py` if needed.
