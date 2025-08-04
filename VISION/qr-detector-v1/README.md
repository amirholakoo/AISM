# QR Code Live Detection System

A high-performance QR code detection system optimized for both Raspberry Pi 5 and desktop computers. This system provides real-time QR code detection with advanced preprocessing techniques and unique code tracking.

## 🚀 Features

- **Real-time QR Detection:**
  - Live video feed processing
  - Multiple preprocessing methods for improved detection
  - Optimized for performance

- **Smart Detection:**
  - Unique QR code tracking
  - Duplicate detection prevention
  - JSON output with metadata

- **Multi-platform Support:**
  - Raspberry Pi 5 optimized (with PiCamera2)
  - Desktop/Laptop compatible
  - Automatic platform detection

- **Advanced Visualization:**
  - Real-time detection overlay
  - FPS monitoring
  - Detection statistics

## 📋 Requirements
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

## 🛠️ Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/amirholakoo/qrcode-detection-system.git
   cd qr-detection-system
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


## 🎯 Usage

### Quick Start
```bash
# Auto-detect platform and run appropriate version
python qr-live-picamera.py

```

### Controls
- **Q** - Quit application
- **S** - Save detections to JSON
- **C** - Clear detection log
- **F** - Toggle fullscreen
- **P** - Toggle preprocessing view (Desktop only)

