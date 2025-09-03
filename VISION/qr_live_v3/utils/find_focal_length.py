import cv2
import time
import argparse
from pyzbar.pyzbar import decode
import numpy as np

# This script uses logic adapted from the main application for camera handling and QR detection.

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False

class CameraStreamer:
    """A simple class to handle video streaming from PiCamera or a standard webcam."""
    def __init__(self, use_picamera=False, video_source=0):
        self.use_picamera = use_picamera
        if self.use_picamera:
            if not PICAMERA2_AVAILABLE:
                raise RuntimeError("PiCamera2 library not found. Please install with: pip install picamera2")
            
            # Use the same advanced configuration as the main application for consistency
            try:
                tuning_file = "imx219_noir.json"
                tuning = Picamera2.load_tuning_file(tuning_file)
                self.picam2 = Picamera2(tuning=tuning)
                print("✅ Loaded camera tuning file 'imx219_noir.json'.")
            except FileNotFoundError:
                print(f"⚠️ WARNING: Tuning file '{tuning_file}' not found. Using default camera settings.")
                self.picam2 = Picamera2()

            config = self.picam2.create_video_configuration(
                main={"size": (1280, 960), "format": "RGB888"},
                raw={'size': (1640, 1232)},
                controls={
                    "FrameDurationLimits": (66666, 66666), # For 15 FPS
                    "ExposureTime": 8000,
                    "AnalogueGain": 4.0
                }
            )
            self.picam2.configure(config)
            self.picam2.start()
            time.sleep(2.0)
        else:
            self.cap = cv2.VideoCapture(video_source)
            time.sleep(2.0)

    def read(self):
        if self.use_picamera:
            frame = self.picam2.capture_array()
            return True, frame
        else:
            return self.cap.read()

    def stop(self):
        if self.use_picamera:
            self.picam2.stop()
        else:
            self.cap.release()

def find_qr_code_width_pixels(frame):
    """Detects the first QR code in a frame and returns its apparent width in pixels."""
    try:
        decoded_objects = decode(frame)
        if decoded_objects:
            main_object = decoded_objects[0]
            points = np.array([p for p in main_object.polygon], dtype=np.int32)
            
            # Use the bounding rectangle to get a stable width measurement
            rect = cv2.minAreaRect(points)
            (w, h) = rect[1]
            
            # The "width" can be either w or h, depending on rotation. We take the larger one.
            apparent_width = max(w, h)
            
            # Draw for visual confirmation
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
            cv2.putText(frame, f"Detected Width: {apparent_width:.2f} px", (box[0][0], box[0][1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
            return apparent_width, frame
    except Exception as e:
        print(f"Error during QR detection: {e}")
    
    return None, frame

def main():
    """Main function to run the focal length calibration."""
    parser = argparse.ArgumentParser(description="Camera Focal Length Calibration Tool")
    parser.add_argument("video_source", type=str, nargs='?', default="0", help="Video source (e.g., camera index like 0).")
    parser.add_argument("--use-picamera", action="store_true", help="Use PiCamera as the video source.")
    args = parser.parse_args()

    print("--- Camera Focal Length Calibration ---")
    print("This tool will help you find the focal length of your camera for distance measurement.")
    print("\nInstructions:")
    print("1.  Measure the REAL width of your QR code paper in centimeters (e.g., 15.0 cm).")
    print("2.  Place the QR code flat against a wall.")
    print("3.  Using a tape measure, position your camera a KNOWN distance from the QR code (e.g., 50.0 cm).")
    print("4.  Ensure the camera is pointing straight at the QR code.")
    print("5.  Enter the values when prompted.\n")

    try:
        known_width_cm = float(input("➡️  Enter the REAL width of the QR code in cm: "))
        known_distance_cm = float(input("➡️  Enter the KNOWN distance to the QR code in cm: "))
    except ValueError:
        print("❌ Invalid input. Please enter numbers only.")
        return

    print("\n🚀 Starting camera... Press 'q' to quit.")
    print("Point the camera at the QR code. The live view will show the detected pixel width.")
    print("When you are ready and the pixel width is stable, press 'c' to calculate the focal length.")

    try:
        video_source = int(args.video_source)
    except ValueError:
        video_source = args.video_source

    streamer = CameraStreamer(use_picamera=args.use_picamera, video_source=video_source)
    
    focal_length = None

    try:
        while True:
            ret, frame = streamer.read()
            if not ret or frame is None:
                print("⚠️ Could not read frame from camera.")
                break

            qr_width_px, display_frame = find_qr_code_width_pixels(frame)

            cv2.imshow("Calibration - Press 'c' to Calculate, 'q' to Quit", display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            
            if key == ord('c'):
                if qr_width_px and qr_width_px > 0:
                    # The formula for focal length
                    focal_length = (qr_width_px * known_distance_cm) / known_width_cm
                    print("\n-------------------------------------------")
                    print(f"✅  Focal Length Calculated!")
                    print(f"   - Detected Pixel Width: {qr_width_px:.2f} px")
                    print(f"   - Known Real Width:   {known_width_cm} cm")
                    print(f"   - Known Distance:     {known_distance_cm} cm")
                    print(f"\n   Your camera's focal length is: {focal_length:.2f}")
                    print("\n   Please copy this value into your config/settings.py file.")
                    print("-------------------------------------------\n")
                    break
                else:
                    print("\n⚠️  Could not calculate. No QR code is visible in the frame. Try again.\n")

    finally:
        streamer.stop()
        cv2.destroyAllWindows()
        print("Application finished.")
        if focal_length:
            print(f"Final calculated focal length: {focal_length:.2f}")

if __name__ == '__main__':
    main() 