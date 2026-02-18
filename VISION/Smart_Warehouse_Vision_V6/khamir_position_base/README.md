# Smart Warehouse Vision - Khamir Position Base

Python-based real-time object detection and tracking pipeline for warehouse monitoring with entry/exit counting.

## Features

- Real-time video processing from `picamera2` or OpenCV capture source
- YOLO detection using Ultralytics
- Kalman filter based object tracking
- Entry/exit and internal displacement counting logic
- JSON output payload generation for integration/API posting

## Project Structure

- `main_khamir.py`: main application and processing pipeline
- `weights/`: YOLO NCNN model directories and metadata
- `imx219_noir.json`, `ov5647_noir.json`: camera-related config files

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

4.  **Install Python Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```




## Requirements

- Python 3.9+
- A valid model path configured in `Config.MODEL_PATH` (default: `weights/best-512_ncnn_model`)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Update settings inside `Config` in `main_khamir.py` as needed:

- `MODEL_PATH`
- `VIDEO_INPUT_PATH` (default: `picamera2`)
- `SHOW_DISPLAY`
- `ENTRY_LINE_X`
- `JSON_FILE_PATH`

## Run

```bash
python main_khamir.py
```

## Notes

- If running on Raspberry Pi, ensure camera stack is installed and configured.
- If `picamera2` is not available, switch input to OpenCV camera or video file in `VIDEO_INPUT_PATH`.
