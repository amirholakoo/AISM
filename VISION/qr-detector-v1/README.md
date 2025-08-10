## Live QR Detection for Raspberry Pi 5 (PiCamera2 / OpenCV)

This project is a lightweight script for live QR code detection on Raspberry Pi 5, which by default uses PiCamera2. If PiCamera2 is not available, it switches to conventional cameras (USB/webcams) via OpenCV.

- Main file: `qr-live-picamera.py`
- Outputs: JSON files in the `results/` folder

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

### Running the Script

```bash
python qr-live-picamera.py
```

**Shortcut keys while running**:
- `q`: Exit
- `s`: Save detections in the `results/` folder
- `c`: Clear current detection log
- `f`: Toggle fullscreen/exit fullscreen

### Outputs
- Upon the first detection of any QR code, it will be logged in memory and saved in a JSON output.
- Output files are created in the `results/` folder, such as:
  - `qr_detections_pi5_YYYYMMDD_HHMMSS.json`
  - `qr_summary_pi5_YYYYMMDD_HHMMSS.json`

### Performance and Image Settings
- The initial resolution is set in the constructor:

```python
QRLiveDetectorPi5(width=1280, height=960, use_picamera2=True)
```

- To increase FPS, reduce the resolution (e.g., 640x480).
- Exposure controls are set in `setup_picamera2`. You can adjust the following values based on lighting conditions:

```python
self.picam2.set_controls({
    "AeEnable": True,        # set to False if you want fully manual exposure
    "ExposureTime": 4000,    # microseconds (e.g., 4000 ≈ 1/250s)
    "AnalogueGain": 8.0
})
```

- If the image is blurry: increase ambient light, reduce `ExposureTime`, or adjust `AnalogueGain`.

### Running on Non-Raspberry Pi Systems
- If PiCamera2 is not available, the script automatically uses `OpenCV VideoCapture(0)`.
- This mode also works for USB webcams on Linux/Windows, but the capabilities and quality may not match PiCamera2.

### Troubleshooting
- Error initializing PiCamera2: Ensure `python3-picamera2` is installed and the camera is enabled in `raspi-config`.
- Error with `imx219_noir.json`: Correct the file path or make it accessible alongside the script, or disable tuning loading.
- Display window does not open: Use `opencv-python-headless` on a server without a graphical environment, or run the app on a desktop.
- Low FPS: Reduce the resolution; the camera frame rate is limited to 30, but you can change this as needed.

### Project Structure
- `qr-live-picamera.py`: Main QR detection code
- `requirements.txt`: Python dependencies
- `results/`: Output folder for JSON files (automatically created)

