# Real-time QR Code Scanner for Industrial Environments

This project provides a robust, high-performance QR code scanning application designed for challenging industrial settings, such as warehouses where cameras are mounted on moving forklifts. It is optimized for real-time video streams and includes features for reliable detection and data logging.

## Key Features

- **High-Speed Detection:** Utilizes the `pyzbar` library for fast and efficient QR code scanning, ensuring high FPS.
- **Robust Video Streaming:** A multi-threaded `VideoStreamer` handles RTSP streams, preventing I/O blocking and ensuring a smooth, real-time video feed.
- **Resilient Connection Handling:** The application includes intelligent connection retries and stream verification to handle network interruptions and ensure the video stream is live before processing.
- **Structured JSON Logging:** Each run of the application generates a unique, timestamped JSON log file containing all unique QR codes detected during the session.
- **Timezone-Aware Timestamps:** All logs include timezone-aware timestamps for accurate record-keeping.
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

## Project Structure

The project is organized into the following directories and files:

- **`robust_live_qr.py`**: The main entry point for the application.
- **`config/`**: Contains the configuration files for the application.
    - **`settings.py`**: The main configuration file where you can adjust parameters like timezone, output directory, and display settings.
- **`core/`**: Contains the core logic of the application.
    - **`video_streamer.py`**: Handles the connection to the video stream.
    - **`qr_scanner.py`**: Manages the QR code detection and processing.
    - **`json_manager.py`**: Manages the creation and saving of JSON log files.
- **`output/`**: The default directory where the timestamped JSON log files are saved.
- **`data/`**: A directory for storing test videos and images.
- **`requirements.txt`**: A list of all the Python libraries required for the project.

## Configuration

All major parameters can be adjusted in the `config/settings.py` file. This includes:

- **`RECONNECT_DELAY_SECONDS`**: The time to wait before attempting to reconnect to a lost video stream.
- **`CONSOLE_LOG_TIMEOUT`**: The cooldown period before printing the same QR code to the console again.
- **`DISPLAY_SCALE`**: A factor to scale the display window for better visibility.
- **`OUTPUT_DIR`**: The directory where the JSON log files are saved.
- **`TIMEZONE`**: The timezone for all timestamps in the logs (defaults to 'Asia/Tehran'). 
