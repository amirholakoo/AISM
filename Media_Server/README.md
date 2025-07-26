
# Raspberry Pi Live Stream Server 🚀

This project sets up a live streaming server on a Raspberry Pi 5 using `mediamtx`. The server is pre-configured to automatically start a camera stream using `rpicam-vid` and `ffmpeg`.

**Reference Links:**
* [Full Tutorial Inspiration](https://github.com/Nerdy-Things/raspberry-pi-5-live-stream/)
* [Raspberry Pi Camera Documentation](https://www.raspberrypi.com/documentation/computers/camera_software.html)
* [Mediamtx GitHub](https://github.com/bluenviron/mediamtx)

---

## ⚙️ Setup Instructions

### 1. Install Dependencies

First, ensure `ffmpeg` is installed on your Raspberry Pi.

```bash
sudo apt update && sudo apt install ffmpeg -y
```

### 2. Install Mediamtx

Next, download and extract the latest ARM64 version of `mediamtx`.

1.  Create a directory for the server.
    ```bash
    mkdir ~/mediamtx
    cd ~/mediamtx
    ```

2.  Go to the [Mediamtx releases page](https://github.com/bluenviron/mediamtx/releases) and copy the download link for the latest `linux_arm64v8.tar.gz` file.

3.  Download and extract it using the link you copied.
    ```bash
    # Replace the URL with the latest version if needed
    wget [https://github.com/bluenviron/mediamtx/releases/download/v1.9.0/mediamtx_v1.9.0_linux_arm64v8.tar.gz](https://github.com/bluenviron/mediamtx/releases/download/v1.9.0/mediamtx_v1.9.0_linux_arm64v8.tar.gz)
    tar -xvzf mediamtx_v1.9.0_linux_arm64v8.tar.gz
    ```

### 3. Apply Custom Configuration 💡

This repository contains a customized `mediamtx.yml` file. **Replace** the default configuration file with the one from this repository.

Assuming you cloned this repository to your home directory (`~`), you can copy it with this command:

```bash
# This command copies our custom .yml file, overwriting the default one.
# Adjust the source path if you cloned the repo elsewhere.
cp ~/your-repo-name/mediamtx.yml ~/mediamtx/mediamtx.yml
```

Our custom configuration automatically starts a stream from the first camera (`--camera 0`) when `mediamtx` launches.

### 4. Run the Media Server

Navigate to the `mediamtx` directory and execute the program.

```bash
cd ~/mediamtx
./mediamtx
```

The server is now running! You'll see log output in your terminal.

### 5. Access the Stream

You can view your live stream at the following addresses from any device on the same network:

* **RTSP:** `rtsp://<RASPBERRY_PI_IP>:8554/cam1`
* **Web Browser (WebRTC/HLS):** `http://<RASPBERRY_PI_IP>:8888/cam1`


