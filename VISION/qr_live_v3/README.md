# Robust QR Code Scanner for Factory & Industrial Environments

This project provides a **robust, high-performance QR code scanning application** specifically engineered for challenging factory and industrial environments. Designed to handle real-world scenarios like **forklift-mounted cameras**, **warehouse operations**, and **moving machinery**, it delivers reliable QR code detection in demanding industrial settings.

## 🏭 Industrial-Grade Features

### **Factory Environment Optimizations**
- **Forklift-Ready:** Handles vibration, movement, and multiple QR codes in warehouse environments
- **Distance-Based Filtering:** Prevents accidental scans of distant codes - only scans intended targets within configurable range (default: 100cm)
- **Motion Stability Validation:** Requires QR codes to be stable for multiple frames, preventing false positives during vehicle movement
- **Multi-QR Code Intelligence:** Automatically focuses on the largest/closest QR code when multiple codes are visible
- **Real-time Distance Display:** Shows exact distance to QR codes for precise targeting and operator feedback

### **Robust Technical Features**
- **High-Speed Detection:** Utilizes the `pyzbar` library for fast and efficient QR code scanning, ensuring high FPS even in industrial conditions
- **Multi-threaded Video Streaming:** Handles RTSP streams with connection retry logic, preventing I/O blocking during network interruptions
- **Advanced Image Processing:** Multiple preprocessing filters (contrast enhancement, noise reduction) for challenging lighting conditions
- **Duplicate Prevention:** Intelligent caching prevents re-logging the same QR code during positioning operations
- **Structured JSON Logging:** Each run generates timestamped logs with all detected QR codes for inventory and audit trails



## Installation

## Setup & Installation

**Do the installation commands in Home directory**

1. **Clone the AISM Repository:**
    ```bash
    git clone https://github.com/amirholakoo/AISM.git
    ```

2.  **Navigate to the Project Directory:**
    ```bash
    cd AISM/VISION/qr_live_v3
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


## Usage

The application is run from the command line and can be configured to use either a PiCamera or a video stream (e.g., RTSP, local file).

**Important note**: When you want to run the app with python command, you must use directly the rasberry terminal not ssh as the UI is not available through ssh.

### Using a PiCamera

For high-performance QR code scanning with a Raspberry Pi camera module, use the `--use-picamera` flag. This leverages the `picamera2` library for direct, efficient camera access.

```bash
python3 robust_live_qr.py --use-picamera
```

### Using an RTSP Stream or Video File

To use an RTSP stream or a local video file, simply provide the path or URL as an argument.

> **Important for RTSP/HTTP Streams:**
> To ensure a stable video stream, you **must use the `mediamtx_v3` server**. Other versions or different streaming servers are not compatible.
>
> Please follow the exact setup instructions from the following repository:
>
> 👉 **[Required: mediamtx_v3 Setup](https://github.com/amirholakoo/AISM/tree/main/Media_Server/mediamtx_v3)**



**Live RTSP Stream:**
```bash
python3 robust_live_qr.py "rtsp://<your-stream-ip>:<port>/<stream-name>"
```
Example:
```bash
python3 robust_live_qr.py rtsp://192.168.237.102:8554/cam1
```

**Local Video File:**
For testing and development, you can also run the application with a local video file:
```bash
python3 robust_live_qr.py "data/test.mp4"
```

### Improving Performance (Headless Mode)

For a significant performance boost, especially on less powerful devices like the Raspberry Pi Zero 2 W, you can run the application in headless mode by disabling the live preview window. This is highly recommended for production use.

To run in headless mode, add the `--no-preview` flag to any of the above commands:
```bash
# Example with PiCamera
python3 robust_live_qr.py --use-picamera --no-preview

# Example with an RTSP stream
python3 robust_live_qr.py rtsp://192.168.237.102:8554/cam1 --no-preview
```


