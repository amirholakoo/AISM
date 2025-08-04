import cv2
import time
import argparse
from config.settings import QRScannerConfig
from core.video_streamer import VideoStreamer
from core.qr_scanner import QRScanner
from core.json_manager import JsonLogManager

def main():
    """Main function to run the QR code scanner."""
    parser = argparse.ArgumentParser(description="Robust Real-time QR Code Scanner for Industrial Use")
    parser.add_argument("video_source", type=str, help="Video source (e.g., RTSP URL, file path, or camera index).")
    args = parser.parse_args()

    config = QRScannerConfig()
    scanner = QRScanner(config)
    streamer = VideoStreamer(args.video_source)
    log_manager = JsonLogManager(config)

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
                fps_frame_count = 0
                fps_start_time = time.time()
            
            fps_text = f"FPS: {display_fps:.2f}"
            cv2.putText(processed_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            h, w, _ = processed_frame.shape
            if h > 0 and w > 0:
                display_frame = cv2.resize(processed_frame, (0, 0), fx=config.DISPLAY_SCALE, fy=config.DISPLAY_SCALE)
                cv2.imshow('Robust Live QR Scanner', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        streamer.stop()
        log_manager.finalize_log()
        cv2.destroyAllWindows()
        print("Application finished.")

if __name__ == '__main__':
    main() 