# Sangin Linebase

A Python-based forklift counting pipeline using YOLO tracking on video input or PiCamera2.

### Hardware Requirements
- Raspberry Pi 5 (preferably 64-bit OS)
- Raspberry Pi camera module (IMX219/IMX477/…)
- Flat cable and proper camera connection

### Software Requirements (Raspberry Pi OS)
- libcamera and PiCamera2
- Python 3.9+ (default in Pi OS)

To install PiCamera2 from the official repositories (recommended):

```bash
sudo apt update
sudo apt install -y python3-picamera2
```

**Note**: Installing PiCamera2 via pip is not always reliable. It is better to use `apt` as mentioned in the README.

### Installing Python Dependencies
1) Create a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
```

2) Install packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

- If you encounter issues installing the OpenCV wheel on Raspberry Pi, you can use the system package:

```bash
sudo apt install -y python3-opencv
```

- If you are on a server without a display, you can use the headless version instead of `opencv-python`:

```bash
pip install opencv-python-headless
```

### Configuration Files (Optional)
The script loads the tuning file `imx219_noir.json`:

```python
# Only used with Camer module V2 noir ( use appropriate configs if needed )
Picamera2.load_tuning_file("imx219_noir.json") 
```

- If this file is not in your default path, either place it alongside the script or provide the full path to the file.
- If the file is missing, you can remove/comment out this section of the code to use the default tuning.


- Python 3.10+ recommended
- Model files available in configured paths (see `Config.MODEL_PATH` in `main.py`)

## Features

- Detects and tracks objects with YOLO
- Counts entry/exit events around a configurable counting line
- Saves snapshots and JSON reports for detected events
- Supports video files and PiCamera2 input

## Project Structure

- `main.py` - main processing loop and reporting
- `config.py` - all runtime configuration
- `video_reader.py` - threaded frame reader
- `object_detector.py` - detection module
- `object_tracker.py` - tracking and counting logic
- `frame_annotator.py` - frame drawing utilities
- `fps_calculator.py` - FPS smoothing helper
- `performance.py` - runtime performance settings
- `weights/` - model files

## Requirements

- Python 3.10+
- Dependencies in `requirements.txt`

Install:

```bash
pip install -r requirements.txt
```

## Run

1. Set your model path and input source in `config.py`.
2. Start the app:

```bash
python main.py
```

## Configuration

Main settings are in `config.py`, including:

- `MODEL_FILENAME`
- `VIDEO_SOURCE`
- `CONFIDENCE_THRESHOLD`
- `FRAME_SKIP`
- `COUNTING_ZONE_MARGIN`
- display/output options

## Output

The app writes outputs to:

- `result/` for snapshots and run report
- `output/logs/` for structured logs
- `temp/` for event JSONL

## Optional Dependencies

If you use Raspberry Pi camera or NCNN-related scripts, install platform-specific packages as needed:

- `picamera2`
- `ncnn`
