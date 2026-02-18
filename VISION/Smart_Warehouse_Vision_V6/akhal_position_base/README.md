# Smart Warehouse Vision - Akhal Position Base

Real-time object detection and tracking pipeline for warehouse vision workflows using YOLO-based inference, OpenCV video processing, and line-crossing event logic.

## Features

- Real-time detection using `ultralytics.YOLO`
- Object tracking with Kalman-based smoothing and ID management
- Line-crossing event counting for classes such as `forklift` and `akhal`
- Works with camera stream (`picamera2`) or video file input
- JSON output support for tracking/event state

## Project Structure

- `main.py` - primary runnable pipeline
- `main_1.py`, `main_2.py`, `main_v1.py`, `main_v2.py` - alternative versions
- `weights/` and other model folders - model assets and NCNN wrappers
- `requirements.txt` - Python dependencies

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

## Configuration

Main runtime settings are in `Config` inside `main.py`, including:

- `MODEL_PATH`
- `VIDEO_INPUT_PATH`
- `ALLOWED_CLASSES`
- `CONFIDENCE_THRESHOLD`
- `ENTRY_LINE_X`
- `JSON_FILE_PATH`

Update these values to match your environment before running.

## Run

```bash
python main.py
```

The app opens a display window when `SHOW_DISPLAY = True`. Press `q` to exit.

## Notes

- If you use NCNN wrapper model files, keep `ncnn` installed.
- `torch` is optional in `main.py` fallback logic, but recommended for best compatibility with `ultralytics`.
- Ensure camera/video source is valid for your system.
