# Smart Warehouse Vision: ONNX-Optimized Forklift Tracking & Counting System

This real-time computer vision application is specifically optimized for high-performance inference using the **ONNX runtime**. It monitors a warehouse environment to track and count forklift movements, utilizing a custom-trained YOLOv11n model to identify forklifts and their loads. By leveraging ONNX, this version achieves significant speed improvements, making it ideal for resource-constrained devices like the Raspberry Pi and for real-time production environments.

The system is designed for robust deployment, featuring a multi-threaded architecture to ensure a smooth user experience even during heavy processing. The user interface is a web-based dashboard built with Streamlit, providing live video feeds, real-time analytics, and post-processing user validation.

This version is optimized for deployment on a **Raspberry Pi** with a connected camera module but is fully capable of running on a standard computer using a video file or any network stream (RTSP/HTTP).

## Key Features

- **High-Performance ONNX Inference**:
    - The core of this version is its use of the ONNX (Open Neural Network Exchange) runtime, which provides faster model loading and inference times compared to standard PyTorch models.
- **Accurate Object Detection**:
    - Utilizes a custom-trained `YOLOv11n` model, converted to the efficient `.onnx` format (`100_epoch_650_yolov11n_v3.onnx`), for high-speed, accurate detection.
- **Region-Based Counting Logic**:
    - Employs a large, configurable central zone for counting, making the system more robust against accidental or repeated line crossings.
- **Live UI Dashboard**: A user-friendly Streamlit interface displays:
    - The live, annotated video feed.
    - Real-time "Loaded" and "Unloaded" counts.
    - Current processing speed (FPS).
    - A live log of all counting events.
- **Manual Correction & Verification**:
    - After a processing session concludes, the system presents a summary of all events.
    - Users can enter a **Manual Edit** workflow to review each detection and correct the classified product type, ensuring 100% data accuracy before final submission.
- **Multi-Platform & Source Support**:
    - **Raspberry Pi**: Native support for the `picamera2` library for efficient, direct camera access.
    - **Standard Systems**: Can process any RTSP/HTTP stream or local video file via OpenCV.
- **Robust & Resilient**:
    - **Multi-Threaded Architecture**: A dedicated thread for frame grabbing and another for processing prevents the UI from freezing and ensures no frames are dropped.
    - **Optimized Inference**: Uses the **ONNX runtime** for efficient, cross-platform model inference, delivering superior performance.
    - **GPU Acceleration**: Automatically utilizes a CUDA-enabled GPU if available, further enhancing the ONNX backend's speed.
    - **Resilient Connectivity**: Automatically retries connecting to a network stream if the connection is unstable.

## Setup & Installation

**Do the installation commands in Home directory**

1. **Clone the AISM Repository:**
    ```bash
    git clone https://github.com/amirholakoo/AISM.git
    ```

2.  **Navigate to the Project Directory:**
    ```bash
    cd AISM/VISION/Smart_Warehouse_Vision_V4
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

## How to Run

1.  **Ensure the Model is Present**:
    Make sure the `100_epoch_650_yolov11n_v3.onnx` file is in the root directory of the project.

2.  **Start the Application**:
    ```bash
    streamlit run main.py
    ```
    This will open the application dashboard in your web browser.

## Using the Application

### Workflow

1.  **Configure Settings**: The counting zone and other parameters are pre-configured in `config/warehouse_config.py`.
2.  **Start Processing**: Enter your video source and click the "▶️ Start Processing" button.
3.  **Monitor Live**: Observe the live feed, counts, and event log. A "SYSTEM READY" message will indicate that the video stream is stable.
4.  **Stop and Review**:
    - Click "⏹️ Stop Processing" to end the session.
    - A summary of the session will be displayed in Persian.
5.  **Confirm or Edit**:
    - **✅ تایید و ذخیره (Confirm & Save)**: If the summary is correct, click this to save the results to a timestamped JSON file.
    - **✏️ ویرایش دستی (Manual Edit)**: If any product types were misidentified, click this to enter the manual correction screen. Here you can change the product type for each event before saving.
