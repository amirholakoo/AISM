# Smart Warehouse Vision V6 - Sangin Position Base

Lightweight object detection and tracking pipeline for warehouse scenarios using YOLO (Ultralytics), OpenCV, and centroid/Kalman-based tracking.

## Setup & Installation

**Do the installation commands in Home directory**

1. **Clone the AISM Repository:**
    ```bash
    git clone https://github.com/amirholakoo/AISM.git
    ```

2.  **Navigate to the Project Directory:**
    ```bash
    cd AISM/VISION/Smart_Warehouse_Vision_V6
    ```

### On a Standard Computer (for Development)

1.  **Create a Virtual Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### On a Raspberry Pi (for Deployment)

Setting up on a Raspberry Pi requires installing system-level dependencies for the camera module.

1.  **Install System Dependencies**:
    These packages are required for `picamera2` to function correctly.
    ```bash
    sudo apt-get update
    sudo apt-get install -y build-essential libcamera-dev python3-libcamera
    ```

2.  **Create the Virtual Environment (with System Access)**:
    This critical step creates a virtual environment that can access the system-level `python3-libcamera` library.
    ```bash
    python3 -m venv --system-site-packages venv
    ```

3.  **Activate the Environment**:
    ```bash
    source venv/bin/activate
    ```

## Features

- Real-time object detection with configurable confidence threshold
- Tracking with Kalman filter smoothing
- Entry/exit counting logic with line-based flow detection
- Optional `picamera2` input support
- JSON-ready output payload for downstream services

## Project Structure

- `main_sangin.py`: Main application logic (capture, detect, track, annotate, output)
- `weights/`: YOLO NCNN model folders
- `requirements.txt`: Python dependencies

## Requirements

- Python 3.9+ (recommended)
- A valid model path in `Config.MODEL_PATH` (default is `weights/best-512_ncnn_model`)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python main_sangin.py
```

## Configuration

Main runtime settings are in `Config` inside `main_sangin.py`, including:

- `VIDEO_INPUT_PATH`
- `MODEL_PATH`
- `CONFIDENCE_THRESHOLD`
- `ENTRY_LINE_X`
- `ALLOWED_CLASSES`

## Notes

- If you run on Raspberry Pi and use `VIDEO_INPUT_PATH = "picamera2"`, install and configure `picamera2` on the device.
- Output payload is posted to the endpoint defined in `main_sangin.py`.
