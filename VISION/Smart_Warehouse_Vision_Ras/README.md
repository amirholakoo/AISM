# Smart Warehouse Vision System

This application uses computer vision to monitor a warehouse environment in real-time. It leverages the YOLOv8 object detection model to identify and track forklifts, automatically counting them as they are loaded or unloaded across a designated line.

This version is optimized for deployment on a **Raspberry Pi** with a connected camera module, but can also run on a standard computer using a video file or network stream.

## Key Features

- **Multi-Platform Support**:
    - **Raspberry Pi**: Utilizes the `picamera2` library for direct, high-performance camera access.
    - **Standard Systems**: Can process any RTSP/HTTP stream or local video file via OpenCV.
- **Live UI Dashboard**: A user-friendly Streamlit interface displays:
    - The live video feed.
    - Real-time "Loaded/Unloaded" counts.
    - Real-time processing speed (FPS).
    - A live log of all counting events.
- **Real-Time Performance Tuning**:
    - **Model Resolution**: Adjust the model's input resolution on the fly to balance accuracy and speed.
    - **Frame Skip**: Change how many frames are skipped to increase FPS.
- **Manual Correction Workflow**: After a processing session, users can manually review and correct the product type for each detected event before saving the final summary.
- **GPU Acceleration**: Automatically utilizes a CUDA-enabled GPU if available on the host machine.
- **Resilient Connectivity**: Automatically attempts to reconnect to a live video stream if the connection is lost.

## Setup & Installation

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

Setting up on a Raspberry Pi requires installing system-level dependencies in addition to the Python packages.

1.  **Install System Dependencies**:
    These packages are required for `picamera2` to build and run correctly.
    ```bash
    sudo apt-get update
    sudo apt-get install -y build-essential libcamera-dev python3-libcamera
    ```

2.  **Create the Virtual Environment (with System Access)**:
    This is the most critical step. We create a `venv` with a special flag that allows it to access the system-level `python3-libcamera` we just installed.
    ```bash
    python3 -m venv --system-site-packages venv
    ```

3.  **Activate the Environment**:
    ```bash
    source venv/bin/activate
    ```

4.  **Install Raspberry Pi Python Dependencies**:
    This requirements file is tailored for the Raspberry Pi's architecture.
    ```bash
    pip install -r requirements_rpi.txt
    ```
    *Note: If you encounter a `numpy` binary incompatibility error upon first run, force a re-installation of the key vision libraries with `pip install --force-reinstall --no-cache-dir numpy opencv-python-headless simplejpeg`.*

## How to Run

1.  **Activate your virtual environment**:
    ```bash
    source venv/bin/activate
    ```

2.  **Start the Application**:
    ```bash
    streamlit run main.py
    ```
    This will open the application dashboard in your web browser.

## Using the Application

- **Production (Live) Tab**:
    - **On Raspberry Pi**: Check the "Use Raspberry Pi Camera" box to use the connected camera module.
    - **On a Standard Computer**: Uncheck the box and enter a network stream URL (RTSP/HTTP) or a path to a local video file.
- **Demo (File) Tab**:
    - This tab runs the system on a pre-packaged local video file (`output_filtered_3.mp4`) for easy testing.
- **Sidebar Controls**:
    - Use the sidebar to adjust the counting line, model resolution, and frame skip rate before starting the processing.
- **Stopping and Saving**:
    - Click "⏹️ Stop Processing" to end a session.
    - You will be presented with a summary of the events.
    - You can choose to "✅ Confirm & Save" the results directly to a timestamped JSON file, or enter the "✏️ Manual Edit" workflow to make corrections before saving.
