# Smart Warehouse Vision - Tolid Position Base

Real-time object detection and counting pipeline for warehouse production flow, built with YOLO and OpenCV.

## Features

- Real-time video processing from `picamera2` or OpenCV video source
- YOLO-based object detection using exported NCNN models
- Kalman-based object tracking with per-class counting logic
- Entry/exit/internal displacement counting
- Optional on-screen visualization with FPS and totals
- JSON output payload with class-wise counters and timestamps

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

## Project Structure

- `main_tolid.py`: Main pipeline (video read, detection, tracking, output)
- `weights/` and model folders: Exported NCNN model assets
- `imx219_noir.json`, `ov5647_noir.json`: Camera-related configuration files

## Requirements

- Python 3.9+
- Dependencies from `requirements.txt`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python main_tolid.py
```

## Configuration

Edit constants in `Config` class inside `main_tolid.py`:

- `MODEL_PATH`: Path to model folder (default: `best-512_ncnn_model`)
- `VIDEO_INPUT_PATH`: `picamera2` or a video file/device id
- `ALLOWED_CLASSES`: Classes used in counting
- `ENTRY_LINE_X`: Vertical line used for movement-based counting
- `JSON_FILE_PATH`: Output path for tracking data
- `SHOW_DISPLAY`: Enable/disable display window
- `CONFIDENCE_THRESHOLD`, `MAX_DETECTIONS`: Detection tuning

## Output

Runtime output includes:

- Per-object tracking state
- `total_entered` by class
- `total_exit` by class
- Current selected `class_name`
- `first_obj_timestamp` and `last_obj_timestamp`

The script can also send data to:

- `http://192.168.2.21:6008/receive_data/`

## Notes

- `picamera2` is typically available on Raspberry Pi systems.
- If you run on desktop/OpenCV source, update `VIDEO_INPUT_PATH` accordingly.
- Make sure model files exist and match `MODEL_PATH`.
