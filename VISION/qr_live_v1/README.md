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
    git clone git@github.com:amirholakoo/AISM.git
    ```

2.  **Navigate to the Project Directory:**
    ```bash
    cd AISM/VISION/qr_live_v1
    ```

3.  **Create a Virtual Environment (Linux):**
    It is recommended to use a virtual environment to manage the dependencies for this project.
    ```bash
    python3 -m venv qrcode_env
    source qrcode_env/bin/activate
    ```

4.  **Install Dependencies:**
    Install all the necessary libraries using the provided `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The application is run from the command line, with the video source provided as an argument.

### Running with a Live RTSP Stream

To connect to a live RTSP stream (e.g., from a camera connected to a MediaMTX server), use the following command:
```bash
python robust_live_qr.py "rtsp://<your-stream-ip>:<port>/<stream-name>"
```
Example:
```bash
python robust_live_qr.py "rtsp://192.168.237.102:8554/cam1"
```

### Running with a Local Video File

For testing and development, you can also run the application with a local video file:
```bash
python robust_live_qr.py "data/test.mp4"
```


