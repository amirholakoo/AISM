# QR Code Live Detection System

A high-performance QR code detection system optimized for Raspberry Pi 5. This system provides real-time QR code detection with advanced preprocessing techniques and unique code tracking.


## Requirements
```txt
opencv-python>=4.8.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
```

### Platform-Specific
- **Raspberry Pi 5:**
  ```bash
  sudo apt install python3-picamera2
  pip install picamera2
  ```

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/amirholakoo/qr-detector-v1.git
   cd qr-detector-v1
   ```

2. **Create Virtual Environment:**
   ```bash
   python -m venv qr-env
   source qr-env/bin/activate  # Linux/Mac
   # OR
   qr-env\Scripts\activate     # Windows
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```


### Quick Start
```bash
# Run the tuned version (uses your custom tuning file if configured)
python qr-live-picamera.py

# Or run the generic version (no tuning file required)
python qr-live-picamera3.py
```