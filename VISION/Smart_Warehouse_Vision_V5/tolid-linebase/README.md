# Paper Roll Counting

Python-based video processing app for counting crossings of detected objects (forklift and paper roll) using YOLO tracking.

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

- `main.py`: main processing pipeline and runtime controls.
- `video_reader.py`: threaded video reader for file/camera input.
- `requirements.txt`: Python dependencies.
- `best_70_new_ncnn_model_with_auggmentation/`: default model directory used by `main.py`.
- `output/`, `result/`, `temp/`: generated during runtime.


## Configuration

Main runtime settings are in `main.py`:

- `MODEL_DIR_NAME`
- `VIDEO_SOURCE`
- `CONF_THRESH`
- `FRAME_SKIP`
- `SHOW_DISPLAY`
- `ENABLE_OUTPUT_VIDEO`

## Run

```bash
python main.py
```

Press `q` to stop when display is enabled.

## Outputs

- Event snapshots: `output/processed/`
- Event log (JSONL): `temp/events_log.json`
- Final report (JSON): `result/`
- Runtime logs: `output/logs/`

## Notes

- If you use Raspberry Pi camera mode, ensure PiCamera2 and camera tuning files are available.
- Default execution is CPU-based.
