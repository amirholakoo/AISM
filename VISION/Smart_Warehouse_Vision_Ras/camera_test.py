import time
from picamera2 import Picamera2

print("--- Starting Camera Test ---")

# Step 1: Check for available cameras
try:
    available_cameras = Picamera2.global_camera_info()
    if not available_cameras:
        print("ERROR: Picamera2 found NO cameras. This is a system-level issue.")
        print("Please ensure the camera is securely connected and has been enabled in the OS configuration.")
    else:
        print(f"SUCCESS: Found the following cameras: {available_cameras}")
except Exception as e:
    print(f"ERROR: An exception occurred while probing for cameras: {e}")
    exit()

# Step 2: Attempt to initialize the default camera
try:
    print("\nAttempting to initialize the default camera...")
    picam2 = Picamera2()
    print("SUCCESS: Camera initialized.")
except Exception as e:
    print(f"ERROR: Failed to initialize the camera: {e}")
    print("This often happens if another application is using the camera or due to a system configuration issue.")
    exit()

# Step 3: Configure and capture an image
try:
    print("\nConfiguring camera for a test capture...")
    config = picam2.create_still_configuration()
    picam2.configure(config)
    print("SUCCESS: Camera configured.")

    print("\nStarting camera...")
    picam2.start()
    time.sleep(2)  # Give the camera time to adjust to light levels

    print("\nCapturing a test image to 'test_image.jpg'...")
    picam2.capture_file("test_image.jpg")
    print("SUCCESS: Image captured and saved as 'test_image.jpg'.")

except Exception as e:
    print(f"ERROR: Failed during capture: {e}")
finally:
    # Step 4: Clean up
    if picam2.started:
        picam2.stop()
        print("\nCamera stopped.")
    print("\n--- Camera Test Finished ---") 