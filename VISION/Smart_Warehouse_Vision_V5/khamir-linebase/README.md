# Khamir Linebase

Real-time object tracking and counting pipeline based on YOLO and OpenCV.

## Features

- Real-time tracking with `ultralytics` YOLO `track()`
- Entry/exit counting on a virtual line
- JSON report and event log generation
- Optional snapshot saving for detected events
- Support for video file input and `picamera2`

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

## Project Structure

- `main.py`: main processing pipeline
- `video_reader.py`: threaded video/picamera reader
- `weights/`: model assets
- `output/processed/`: saved event snapshots
- `output/logs/`: generated run logs
- `temp/events_log.json`: event JSON lines log

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Update values in `main.py` based on your environment:

- `MODEL_DIR_NAME`
- `VIDEO_SOURCE`
- `SHOW_DISPLAY`
- `FRAME_SKIP`
- `CONF_THRESH`

Environment variables:

- `ANBAR_THREADS` (default: `4`)
- `ANBAR_EXPOSURE_US` (default: `16000`)
- `ANBAR_ANALOG_GAIN` (default: `7.5`)
- `ANBAR_DIGITAL_GAIN` (optional)

## Run

```bash
python main.py
```

Press `q` to stop the display window.

## Outputs

- Report JSON files are written to `result/`
- Event logs are written to `temp/events_log.json`
- Runtime logs are written to `output/logs/` and `log.txt`
