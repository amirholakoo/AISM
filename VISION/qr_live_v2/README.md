# Real-time QR Code Scanner for Industrial Environments

This project provides a robust, high-performance QR code scanning application designed for challenging industrial settings, such as warehouses where cameras are mounted on moving forklifts. It is optimized for real-time video streams and includes features for reliable detection and data logging.

## Key Features

- **High-Speed Detection:** Utilizes the `pyzbar` library for fast and efficient QR code scanning, ensuring high FPS.
- **Robust Video Streaming:** A multi-threaded `VideoStreamer` handles RTSP streams, preventing I/O blocking and ensuring a smooth, real-time video feed.
- **Resilient Connection Handling:** The application includes intelligent connection retries and stream verification to handle network interruptions and ensure the video stream is live before processing.
- **Structured JSON Logging:** Each run of the application generates a unique, timestamped JSON log file containing all unique QR codes detected during the session.
- **Modular and Configurable:** The project is organized into a clean, modular structure with a central configuration file, making it easy to adapt and extend.

## Installation

1. **Clone the AISM Repository:**
    ```bash
    git clone https://github.com/amirholakoo/AISM.git
    ```

2.  **Navigate to the Project Directory:**
    ```bash
    cd AISM/VISION/qr_live_v1
    ```

3.  **Create a Virtual Environment (Linux):**
    It is recommended to use a virtual environment to manage the dependencies for this project.
    ```bash
    python3 -m venv --system-site-packages qrcode_env 
    source qrcode_env/bin/activate
    ```

4.  **Install Dependencies:**
    Install all the necessary libraries using the provided `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

    For PiCamera support, you will also need to install the `picamera2` library:
    ```bash
    pip install picamera2
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


