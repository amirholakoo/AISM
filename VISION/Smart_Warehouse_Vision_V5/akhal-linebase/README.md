# Akhal Linebase

Video-based object detection and counting project for warehouse workflows.

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

Install dependencies:

```bash
pip install -r requirements.txt
```

## Features
- YOLO-based detection pipeline
- Kalman/simple tracking modes
- Entry/exit counting across virtual lines
- Snapshot and event logging support
- Support for video file input and optional PiCamera2 input

## Project Structure
- `main.py`: main counting Akhal pipeline for Anbar_khamir_kordan
- `main-khamir.py`: khamir-specific pipeline
- `main-tolid.py`: tolid-specific pipeline
- `main-kalman.py`: alternative tracking/counter implementation
- `video_reader.py`: threaded video reader (file/camera)
- `weights*/.../model_ncnn.py`: generated NCNN model wrappers

## Requirements
Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage
1. Put your input video in `video_source/pending/` or update the input path in the script config.
2. Ensure your model path is correct (`weights/...` or other configured path).
3. Run one of the main scripts:

```bash
python main.py
```

Or:

```bash
python main-kalman.py #for anbar_khamir_kordan
```
```bash
python main-khamir.py #for anbar_khamir
```
```bash
python main-tolid.py #for anbar_tolid
```

## Outputs
- `output/processed/`: snapshots
- `output/logs/`: logs
- `temp/events_log.json`: event stream
- `result/`: optional final report files

## Notes
- `picamera2` is optional and intended for Linux/Raspberry Pi environments.
- If a specific model path is missing, update the corresponding config values in the script you run.
