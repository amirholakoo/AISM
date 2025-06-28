# Smart Warehouse Vision System

This application uses computer vision to monitor a warehouse environment in real-time. It leverages the YOLOv8 object detection model to identify and track forklifts, automatically counting them as they are loaded or unloaded across a designated line.

## Key Features

- **Real-Time Processing**: Connects to any RTSP camera stream for live, 24/7 monitoring of warehouse activity.
- **Demo Mode**: Allows for easy testing and validation of the detection model using a local video file.
- **GPU Acceleration**: Automatically utilizes a CUDA-enabled GPU for high-performance, high-accuracy video analysis.
- **Event Logging**: Every "loaded" and "unloaded" event is logged to a persistent `warehouse_events.log` file for auditing.
- **Event Snapshots**: Captures and saves a tagged image of every single counting event to the `snapshots/` directory, providing visual proof.
- **Resilient Connectivity**: Automatically attempts to reconnect to the live video stream if the connection is lost.
- **Live UI Dashboard**: A user-friendly Streamlit interface displays the live video feed, real-time statistics, and a live log of all counting events.
- **Manual Correction**: A post-session workflow allows users to manually review and correct any detection if needed.

## Installation

To get the application up and running, follow these steps.

1.  **Set up a Python Environment**: It is highly recommended to use a virtual environment to avoid conflicts with other projects.

    ```bash
    # Create the virtual environment
    python3 -m venv venv

    # Activate it (on Linux/macOS)
    source venv/bin/activate

    # On Windows, use:
    # venv\Scripts\activate
    ```

2.  **Install Dependencies**: Install all required Python packages using the `requirements.txt` file.

    ```bash
    pip install -r requirements.txt
    ```

## How to Run and Use

Once the installation is complete, you can run the application with a single command.

1.  **Start the Application**:

    ```bash
    streamlit run main.py
    ```

    This will open the application dashboard in your web browser.

2.  **Using the Tabs**:

    -   **Production (Live)**: This is the main tab for real-world use.
        -   Enter the RTSP URL of your camera (e.g., `rtsp://192.168.1.101:8554/mystream`).
        -   Click "▶️ Start Processing" to begin live monitoring.

    -   **Demo (File)**: This tab is for testing the system.
        -   **Important**: You must first download the demo video file (`output_filtered_3.mp4`) from [this link](https://drive.google.com/file/d/1aX3nUsxQ-xzsFmH9okSezTpEvXDs7_iB/view?usp=sharing).
        -   Place the downloaded file in the main project directory.
        -   Once the file is in place, simply click "▶️ Start Processing" to see how the system works.

3.  **Monitoring the System**:

    -   While running, the live video feed will be displayed on the left.
    -   On the right, you will see the running counts and a live table of all crossing events.
    -   After stopping a session, you will be presented with a summary and the option to save the results to a final JSON file.

4.  **Checking Logs and Snapshots**:
    -   All detailed logs are saved in the `warehouse_events.log` file in the main project directory.
    -   All event-triggered images are saved in the `snapshots/` directory.
