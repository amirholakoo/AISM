import cv2
import time
import argparse
from config.settings import QRScannerConfig
from core.video_streamer import VideoStreamer
from core.qr_scanner import QRScanner
from core.json_manager import JsonLogManager

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False

class PiCameraStreamer:
    """Class to handle PiCamera-specific streaming."""
    def __init__(self, width=1280, height=960, framerate=15):
        if not PICAMERA2_AVAILABLE:
            raise RuntimeError("PiCamera2 library is not installed. Please install with: pip install picamera2")
        tuning_file = "imx219_noir.json"
        tuning = Picamera2.load_tuning_file(tuning_file)
        self.picam2 = Picamera2(tuning=tuning)

        #tuning = self.picam2.load_tuning_file(tuning_file)
        config = self.picam2.create_video_configuration(
                    main={"size": (1280, 960), "format": "RGB888"},
                    raw={'size': (1640, 1232)},
                    controls={
                        "FrameDurationLimits": (66666, 66666), # For 15 FPS
                        "ExposureTime": 8000,      # Corresponds to --shutter 8000
                        "AnalogueGain": 4.0        # Corresponds to --gain 4
                    }
                )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(2)  # Allow camera to warm up

    def read(self):
        """Capture a frame from the camera."""
        frame = self.picam2.capture_array()
        return True, frame

    def stop(self):
        """Stop the camera stream."""
        self.picam2.stop()
        
    def connect(self):
        """Dummy connect method to match VideoStreamer interface."""
        return True

def main():
    """Main function to run the QR code scanner."""
    parser = argparse.ArgumentParser(description="Robust Real-time QR Code Scanner for Industrial Use")
    parser.add_argument("video_source", type=str, nargs='?', default=None, help="Video source (e.g., RTSP URL, file path, or camera index). Required if --use-picamera is not set.")
    parser.add_argument("--use-picamera", action="store_true", help="Use PiCamera as the video source.")
    parser.add_argument("--no-preview", action="store_true", help="Disable the live preview window to improve performance.")
    args = parser.parse_args()

    if args.use_picamera and not PICAMERA2_AVAILABLE:
        print("❌ --use-picamera was specified, but PiCamera2 library is not available.")
        print("   Please install it using: pip install picamera2")
        return
        
    if not args.use_picamera and args.video_source is None:
        parser.error("A video_source is required when --use-picamera is not specified.")

    config = QRScannerConfig()
    scanner = QRScanner(config)
    log_manager = JsonLogManager(config)

    if args.use_picamera:
        print("🚀 Initializing PiCamera...")
        streamer = PiCameraStreamer(
            width=config.CAMERA_RESOLUTION[0], 
            height=config.CAMERA_RESOLUTION[1],
            framerate=config.FRAME_RATE
        )
    else:
        print(f"🚀 Initializing video stream from: {args.video_source}")
        streamer = VideoStreamer(args.video_source, config)

    print("🚀 Starting robust QR scanner... Press 'q' to quit.")

    if not streamer.connect():
        print("❌ Could not connect to video source. Exiting.")
        log_manager.finalize_log()
        return

    fps_start_time = time.time()
    fps_frame_count = 0
    display_fps = 0

    try:
        while True:
            ret, frame = streamer.read()
            
            if not ret or frame is None:
                print("⚠️ Main loop: Could not read frame. Stream may have ended.")
                time.sleep(config.RECONNECT_DELAY_SECONDS)
                continue
                
            processed_frame, qr_data = scanner.process_frame(frame)
            
            if qr_data:
                log_manager.add_record(qr_data)
            
            fps_frame_count += 1
            if time.time() - fps_start_time >= 1.0:
                display_fps = fps_frame_count / (time.time() - fps_start_time)
                # Print FPS to the terminal, overwriting the previous line
                print(f"Processing FPS: {display_fps:.2f}", end="\r")
                fps_frame_count = 0
                fps_start_time = time.time()
            
            fps_text = f"FPS: {display_fps:.2f}"
            cv2.putText(processed_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            h, w, _ = processed_frame.shape
            if h > 0 and w > 0 and not args.no_preview:
                display_frame = cv2.resize(processed_frame, (0, 0), fx=config.DISPLAY_SCALE, fy=config.DISPLAY_SCALE)
                cv2.imshow('Robust Live QR Scanner', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        print()  # Print a newline to move to the next line in the terminal on exit
        streamer.stop()
        log_manager.finalize_log()
        cv2.destroyAllWindows()
        print("Application finished.")

if __name__ == '__main__':
    main() 
