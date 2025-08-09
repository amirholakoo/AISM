
# Raspberry Pi Live Stream Server 🚀

This project sets up a live streaming server on a Raspberry Pi 5 using `mediamtx`. The server is pre-configured to automatically start a camera stream using `rpicam-vid` and `ffmpeg`.

**Reference Links:**
* [Full Tutorial Inspiration](https://github.com/Nerdy-Things/raspberry-pi-5-live-stream/)
* [Raspberry Pi Camera Documentation](https://www.raspberrypi.com/documentation/computers/camera_software.html)
* [Mediamtx GitHub](https://github.com/bluenviron/mediamtx)

---

## ⚙️ Setup Instructions

### 1. Clone This Repository

First, clone this repository to your home directory.

```bash
git clone https://github.com/amirholakoo/AISM.git
```
```bash
cd AISM/Media_Server/mediamtx_v3
```
*Replace `<URL_OF_YOUR_REPOSITORY>` with the actual Git URL and `your-repo-name` with the directory name.*

### 2. Install Dependencies

Next, ensure `ffmpeg` is installed on your Raspberry Pi.

```bash
sudo apt update && sudo apt install ffmpeg -y
```

### 3. Install Mediamtx

Now, download and extract the latest ARM64 version of `mediamtx`.

1.  Create a directory for the server.
    ```bash
    mkdir ~/mediamtx
    cd ~/mediamtx
    ```

2.  Go to the [Mediamtx releases page](https://github.com/bluenviron/mediamtx/releases) and copy the download link for the latest `mediamtx_v..._linux_arm64.tar.gz` file.

3.  Download and extract it using the link you copied.
    ```bash
    # Replace the URL with the latest version if needed
    wget https://github.com/bluenviron/mediamtx/releases/download/v1.13.1/mediamtx_v1.13.1_linux_arm64.tar.gz
    ```
    ```bash
    tar -xvzf mediamtx_v1.13.1_linux_arm64.tar.gz
    ```

### 4. Apply Custom Configuration 💡

You now need to copy the configuration and tuning files from this repository into the `~/mediamtx` directory you just created. You can do this automatically with the command below or manually using a file explorer.

**Option A: From the command line (recommended)**
*Make sure you are inside the cloned repository directory before running this.*
```bash
# This command copies our custom .yml file, overwriting the default one.
# It also copies the camera tuning files.
cp ~/AISM/Media_Server/mediamtx_v3/ov5647_noir.json ~/mediamtx
cp ~/AISM/Media_Server/mediamtx_v3/imx219_noir.json ~/mediamtx
cp ~/AISM/Media_Server/mediamtx_v3/mediamtx.yml ~/mediamtx
```

**Option B: Manually**
1. Open your file explorer.
2. Navigate to the directory where you cloned this repository.
3. Copy the `mediamtx.yml`, `imx219_noir.json`, and `ov5647_noir.json` files.
4. Paste them into the `~/mediamtx` folder, overwriting the existing `mediamtx.yml` file.

Our custom configuration automatically starts a stream from the first camera (`--camera 0`) when `mediamtx` launches.

### 5. Run the Media Server

Navigate to the `mediamtx` directory and execute the program.

> **Important:** You must run the `./mediamtx` command from within the `~/mediamtx` directory. This ensures that the relative paths in `mediamtx.yml` for the camera tuning files are found correctly.

```bash
cd ~/mediamtx
./mediamtx
```

The server is now running! You'll see log output in your terminal.

### 6. Access the Stream

You can view your live stream at the following addresses from any device on the same network:

* **RTSP:** `rtsp://<RASPBERRY_PI_IP>:8554/cam1`
* **Web Browser (WebRTC/HLS):** `http://<RASPBERRY_PI_IP>:8888/cam1`


