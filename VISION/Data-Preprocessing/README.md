# Sack Detection using YOLOv8-OBB - Complete Step-by-Step Guide

This project provides a complete pipeline for training and deploying an object detection model with Oriented Bounding Boxes (OBB) using YOLOv8. The primary goal is to detect sacks in images.

**This guide is designed for complete beginners** - follow every step carefully and you'll have a working sack detection system.

## Prerequisites

- A Raspberry Pi with camera module
- A Windows laptop/PC  
- Internet connection
- Basic computer skills (copy/paste, download files)

---

## Step 1: Connect to Raspberry Pi and Record Videos

### 1.1 Connect to Raspberry Pi via SSH

1. **Find your Raspberry Pi's IP address** (ask your network administrator or check your router's connected devices)
2. **Open Command Prompt** on Windows:
   - Press `Windows + R`
   - Type `cmd` and press Enter
3. **Connect via SSH**:
   ```bash
   ssh admin@YOUR_RPI_IP_ADDRESS
   ```
   Replace `YOUR_RPI_IP_ADDRESS` with the actual IP (e.g., `172.16.6.73`)
4. **Enter the password** when prompted (default: `pi`)

### 1.2 Record Videos

1. **Navigate to the recording script**:
   ```bash
   cd /home/admin
   ls
   ```
   You should see `record_and_stream_wpar.py` in the list

2. **Run the recording script**:
   ```bash
   python3 record_and_stream_wpar.py
   ```

3. **Record videos from different angles**:
   - Move the camera to capture sacks from various positions
   - Record for 30-60 seconds per angle
   - Press `Ctrl+C` to stop recording
   - Repeat for different angles (at least 5-10 different positions)

4. **Check recorded videos**:
   ```bash
   ls -la videos/
   ```
   You should see your recorded video files in the `videos` folder

---

## Step 2: Transfer Videos to Your Computer

### 2.1 Install FileZilla (Windows)

1. **Download FileZilla**:
   - Go to: https://filezilla-project.org/download.php?type=client
   - Click "Download FileZilla Client"
   - Choose "FileZilla_3.x.x_win64-setup.exe" for 64-bit Windows

2. **Install FileZilla**:
   - Run the downloaded installer
   - Click "Next" through all steps
   - Click "Install"
   - Click "Finish"

### 2.2 Transfer Videos Using FileZilla

1. **Open FileZilla**
2. **Connect to Raspberry Pi**:
   - Host: `YOUR_RPI_IP_ADDRESS` (same as SSH)
   - Username: `admin`
   - Password: `pi` (or your Pi's password)
   - Port: `22`
   - Click "Quickconnect"

3. **Navigate on Raspberry Pi side** (right panel):
   - Go to `/home/admin/videos`
   - You should see your `.mp4` files in the videos folder

4. **Create folder on your computer** (left panel):
   - Navigate to your Desktop
   - Right-click → "Create directory"
   - Name it "sack_videos"

5. **Transfer videos**:
   - Select all `.mp4` files on the right panel
   - Drag them to the "sack_videos" folder on the left panel
   - Wait for transfer to complete

---

## Step 3: Video Cropping (Optional)

If your videos contain unnecessary areas (like walls, irrelevant objects), crop them:

1. **Download LosslessCut**:
   - Go to: https://github.com/mifi/lossless-cut/releases
   - Download "LosslessCut-win-x64.exe"
   - Run the exe file

2. **Crop videos**:
   - Open each video in LosslessCut
   - Use the crop tool to remove irrelevant areas
   - Export the cropped video
   - Save cropped videos in a new folder "cropped_videos"

---

## Step 4: Install Docker and CVAT

### 4.1 Install Docker Desktop

1. **Download Docker Desktop**:
   - Go to: https://www.docker.com/products/docker-desktop/
   - Click "Download for Windows"
   - Run the installer `Docker Desktop Installer.exe`

2. **Install Docker**:
   - Follow installation wizard
   - **Important**: Enable WSL 2 when asked
   - Restart your computer when installation completes

3. **Verify Docker is running**:
   - Look for Docker whale icon in system tray (bottom-right corner)
   - Open Command Prompt and run:
     ```bash
     docker --version
     ```
   - You should see Docker version information

### 4.2 Install CVAT

1. **Download CVAT**:
   ```bash
   git clone https://github.com/opencv/cvat
   cd cvat
   ```
   
   **If you don't have Git installed**:
   - Download Git from: https://git-scm.com/download/win
   - Install it, then run the above commands in Command Prompt

2. **Start CVAT**:
- run this command in ubuntu (wsl2) in terminal
   ```bash
   cd cvat
   docker compose up -d
   ```
   Wait 5-10 minutes for all services to start

3. **Access CVAT**:
   - Open your web browser
   - Go to: `http://localhost:8080`
   - You should see CVAT login page

---

## Step 5: Label Videos in CVAT

### 5.1 Create CVAT Account

1. **Register new account**:
   - Click "Create an account"
   - Fill in details (username: `admin`, email: `admin@example.com`)
   - Click "Submit"

### 5.2 Create New Task

1. **Click "Tasks" in top menu**
2. **Click "Create new task"**
3. **Fill task details**:
   - Name: `Sack Detection Task 1`
   - Labels: Click "Add label" → Type `sack` → Click "continue"
   - **Important**: Set "Bug tracker" to empty (delete default text)

4. **Upload videos**:
   - Click "Select files"
   - Choose your video files from "sack_videos" or "cropped_videos" folder
   - Click "Submit & Open"

### 5.3 Label the Sacks

1. **Open the task** by clicking on it
2. **Start labeling**:
   - Click "Job #1" to start annotation
   - **Select "Rectangle" tool** from left toolbar
   - **For each sack in the video**:
     - Draw a rectangle around the sack
     - **Important**: After drawing, you can rotate the rectangle by dragging the rotation handle
     - Make sure the rotated rectangle fits tightly around the sack
     - Label will automatically be "sack"

3. **Navigate through frames**:
   - Use spacebar to play/pause
   - Use arrow keys to go frame by frame
   - Label sacks in every frame where they appear

4. **Track objects across frames**:
   - After labeling a sack in one frame, CVAT can track it automatically
   - Right-click on the labeled box → "Track"
   - Review and adjust tracking as needed

### 5.4 Export Annotations

1. **Go back to Tasks page** (click "Tasks" in top menu)
2. **Click on your task name**
3. **Click "Actions" → "Export task dataset"**
4. **Select format**: "CVAT for video 1.1"(click on bottom Save Image)
5. **Click "OK"**
6. **Download the ZIP file** when ready
7. **Extract the ZIP file** to a folder called "cvat_annotations"

---

## Step 6: Prepare Development Environment

### 6.1 Install Python and Required Tools

1. **Download Python 3.9 or 3.10**:
   - Go to: https://www.python.org/downloads/
   - Download Python 3.10.x
   - **Important**: Check "Add Python to PATH" during installation

2. **Install Git** (if not already installed):
   - Go to: https://git-scm.com/download/win
   - Download and install

### 6.2 Download Project Files

1. **Create project folder**:
   Create New Folder and Rename it "sack_detection"
   or
   ```bash
   mkdir C:\sack_detection
   cd C:\sack_detection
   ```

2. **Copy all Python scripts**  and go to this folder:
   - `convert_to_yolov8_obb.py`
   - `clean_and_renumber_dataset.py`
   - `train_yolov8_obb.py`
   - `dataset_stats.py`
   - `requirements.txt`

### 6.3 Create Python Virtual Environment

1. **Create virtual environment**:(open windows terminal)

   ```bash
   cd C:\sack_detection
   python -m venv env
   ```

2. **Activate virtual environment**:
   ```bash
   venv\Scripts\activate
   ```
   You should see `(env)` at the beginning of your command prompt

3. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   or 
   pip install ultralytics opencv-python pyyaml numpy
   ```

---

## Step 7: Convert CVAT Annotations to YOLO Format

### 7.1 Prepare Files

1. **Copy annotation files**:
   - From your "cvat_annotations" folder, copy `annotations.xml` and "images" to `C:\sack_detection\`

### 7.2 Run Conversion Script

1. **Check file names** in `convert_to_yolov8_obb.py`:
   ```python
   # Line ~306-308, make sure these match your files:
   xml_path = "annotations.xml"
   images_dir = "images"
   output_dir = "yolov8_obb_dataset"
   ```

2. **Run the conversion**:
   ```bash
   python convert_to_yolov8_obb.py
   ```

3. **Verify output**:
   - You should see a new folder `yolov8_obb_dataset`
   - Inside should be `train` and `val` folders
   - Each containing `images` and `labels` subfolders

---

## Step 8: Clean and Prepare Dataset

1. **Run dataset cleaning**:
   ```bash
   python clean_and_renumber_dataset.py
   ```

2. **Check cleaned dataset**:
   - New folder `yolov8_obb_cleaned_dataset` should be created
   - This removes images without annotations and renumbers files

3. **View dataset statistics**:
   ```bash
   python dataset_stats.py
   ```

---

## Step 9: Train the Model

### 9.1 Download Pre-trained Model

1. **Download YOLOv8-OBB model**:
   ```bash
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-obb.pt
   ```
   
   **If wget doesn't work**:
   - Go to: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-obb.pt
   - Save the file to `C:\sack_detection\`

### 9.2 Configure Training Parameters

1. **Edit `train_yolov8_obb.py`** if needed:
   - Line ~189: `dataset_path = "yolov8_obb_cleaned_dataset"`
   - Line ~190: `epochs = 50` (adjust if needed)
   - Line ~191: `model_size = 'n'` (n=nano, s=small, m=medium, l=large)

### 9.3 Start Training

1. **Run training**:
   ```bash
   python train_yolov8_obb.py
   ```

2. **Monitor progress**:
   - Training will show progress bars and metrics
   - Results will be saved in `yolov8_obb_training\sack_detection\`
   - Best model will be saved as `weights\best.pt`

---

## Step 10: Test Your Model
you can get a test image in val images in "sack_detection\yolov8_obb_cleaned_dataset\val\"
1. **Test the trained model**:
   ```bash
   yolo obb predict model=yolov8_obb_training\sack_detection\weights\best.pt source=path\to\test\image.png
   ```

2. **Check results**:
   - Predictions will be saved in `runs\obb\predict\`
   - Open the images to see detected sacks with oriented bounding boxes

---

## Troubleshooting

### Common Issues:

1. **"Permission denied" errors**: Run Command Prompt as Administrator
2. **"Module not found"**: Make sure virtual environment is activated
3. **Docker not starting**: Restart Docker Desktop from system tray
4. **CVAT not loading**: Wait longer, or restart Docker containers
5. **Training fails**: Check if you have enough GPU memory, reduce batch size

### File Structure Check:

Your final folder should look like this:
```
C:\sack_detection\
├── annotations.xml
├── images/
├── yolov8_obb_dataset/
├── yolov8_obb_cleaned_dataset/
├── yolov8_obb_training/
├── convert_to_yolov8_obb.py
├── clean_and_renumber_dataset.py
├── train_yolov8_obb.py
├── dataset_stats.py
├── yolov8n-obb.pt
└── venv/
```

---

## Need Help?

If you encounter any issues:
1. Check the file paths and names match exactly
2. Ensure all files are in the correct locations
3. Make sure virtual environment is activated
4. Verify Docker is running (for CVAT steps)
5. Check that all required files were downloaded/copied correctly

**Remember**: Each step builds on the previous one, so complete them in order!
