# YOLOe Detector

YOLOe Detector is a real-time object detection system designed for detecting sacks using YOLO models on Raspberry Pi with Picamera2. It supports oriented bounding boxes (OBB) for improved accuracy in detecting rotated objects. The system uses multi-threading to optimize performance, separating frame capture and detection processes.

## Features
- Real-time sack detection using YOLOv8 OBB models
- Multi-threaded architecture for improved FPS
- FPS calculation and display
- Configurable confidence threshold
- Oriented bounding box visualization

## Requirements
- Raspberry Pi (tested on models with PiSP support)
- Picamera2 compatible camera (e.g., IMX219)
- Python 3.9+
- Libraries:
  - torch
  - opencv-python (cv2)
  - numpy
  - picamera2
  - ultralytics (for YOLO)
  - Other dependencies as imported in the scripts

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

**Note**: Installing PiCamera2 via pip is not always reliable. It is better to use `apt`.

### Installing Python Dependencies
1) Clone the repository or copy the scripts to your Raspberry Pi:
   ```
   git clone https://github.com/amirholakoo/AISM.git

   cd AISM/VISION/yolo-sack-detector
   ```

2) Create a virtual environment (recommended):
   ```
   python3 -m venv .env
   source .venv/bin/activate  # on Windows: .env\Scripts\activate
   ```

3) Install packages:
   ```
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

- If you encounter issues installing the OpenCV wheel on Raspberry Pi, you can use the system package:
   ```
   sudo apt install -y python3-opencv
   ```

### On a Raspberry Pi (for Deployment)

Setting up on a Raspberry Pi requires installing system-level dependencies for the camera module.

1) **Install System Dependencies**:
These packages are required for picamera2 to function correctly.

sudo apt-get update
sudo apt-get install -y build-essential libcamera-dev python3-libcamera

2) **Create the Virtual Environment (with System Access)**:
This critical step creates a virtual environment that can access the system-level python3-libcamera library.

python3 -m venv --system-site-packages venv

3) **Activate the Environment**::

source venv/bin/activate

4) **Install Python Dependencies**:

pip install -r requirements.txt

## Usage

### Detection Script (detect-sacks-v2.py)

This script performs real-time detection of sacks.

Run in headless mode:
```
python detect-sacks-v2.py --headless
```

- In headless mode, it processes a fixed number of frames (configurable in code).

### Tracking Script (track-sack.py)

This script extends detection with tracking capabilities.

Run:
```
python track-sack.py
```

## Configuration

- **Model Path**: Set in `__init__` (default: 'weights/best.pt')
- **Resolution**: Configured in camera setup (default: 960x720)
- **Confidence Threshold**: Set in `__init__` (default: 0.5)
- **Headless Mode**: Pass `headless=True` to the detector class

Modify these in the script as needed.

## How It Works

1. **Initialization**:
   - Sets up Picamera2 with specified resolution.
   - Loads YOLO model.

2. **Multi-threading**:
   - Main thread: Captures frames, queues copies for detection, displays results using latest detections.
   - Detection thread: Processes queued frames and updates shared detections.

3. **Detection**:
   - Uses YOLO to detect objects, supporting both OBB and regular boxes.
   - Draws bounding boxes and FPS on the frame.

4. **FPS Calculation**: Updates every second based on processed frames.




