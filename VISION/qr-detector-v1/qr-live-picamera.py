import cv2
import numpy as np
import os
import pandas as pd
import time
import json
from datetime import datetime

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    print("⚠️  PiCamera2 not available. Install with: pip install picamera2")
    print("    Falling back to OpenCV VideoCapture")

# Set paths for optional logging
OUTPUT_PATH = 'results'
os.makedirs(OUTPUT_PATH, exist_ok=True)

class QRLiveDetectorPi5:
    def __init__(self, width=1280, height=720, use_picamera2=True):
        """Initialize the live QR detector for Raspberry Pi 5"""
        self.width = width
        self.height = height
        self.detector = cv2.QRCodeDetector()
        self.detected_qr_codes = []
        self.unique_qr_codes = set()  # Keep track of unique QR codes
        self.use_picamera2 = use_picamera2 and PICAMERA2_AVAILABLE
        
        # Initialize camera
        self.picam2 = None
        self.cap = None
        self.setup_camera()
        
    def setup_camera(self):
        """Setup camera for Raspberry Pi 5"""
        if self.use_picamera2:
            return self.setup_picamera2()
        else:
            return self.setup_opencv_camera()
    
    def setup_picamera2(self):
        """Setup using PiCamera2 (recommended for Pi 5)"""
        try:
            self.picam2 = Picamera2()
            
            # Configure camera
            config = self.picam2.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "BGR888"}
            )
            self.picam2.configure(config)
            self.picam2.start()
            
            # Allow camera to warm up
            time.sleep(2)
            
            print(f"✅ PiCamera2 initialized successfully - Resolution: {self.width}x{self.height}")
            return True
            
        except Exception as e:
            print(f"❌ PiCamera2 initialization failed: {e}")
            print("Falling back to OpenCV...")
            return self.setup_opencv_camera()
    
    def setup_opencv_camera(self):
        """Fallback to OpenCV VideoCapture"""
        try:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                raise Exception("OpenCV camera test failed")
                
            print(f"✅ OpenCV Camera initialized - Resolution: {self.width}x{self.height}")
            return True
            
        except Exception as e:
            print(f"❌ Camera initialization failed: {e}")
            return False
    
    def capture_frame(self):
        """Capture frame from camera"""
        if self.picam2:
            # PiCamera2 capture
            frame = self.picam2.capture_array()
            # Convert RGB to BGR for OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return True, frame
        elif self.cap:
            # OpenCV capture
            return self.cap.read()
        else:
            return False, None
    
    def extract_qr_content_optimized(self, frame):
        """Fast QR code extraction optimized for Pi 5"""
        if frame is None:
            return None, frame, None
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Fast preprocessing methods (limited for real-time performance)
        preprocessing_methods = [
            ("Original", frame),
            ("Otsu", cv2.cvtColor(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1], cv2.COLOR_GRAY2BGR)),
            ("CLAHE", cv2.cvtColor(cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(gray), cv2.COLOR_GRAY2BGR)),
            ("Adaptive", cv2.cvtColor(cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2), cv2.COLOR_GRAY2BGR)),
        ]
        
        # Try each method quickly
        for method_name, processed_img in preprocessing_methods:
            try:
                qr_data, bbox, confidence = self.detector.detectAndDecode(processed_img)
                
                if qr_data and len(qr_data.strip()) > 0:
                    # Create visualization
                    vis_img = frame.copy()
                    if bbox is not None and len(bbox) > 0:
                        bbox = bbox.astype(int)
                        cv2.polylines(vis_img, [bbox], True, (0, 255, 0), 2)
                        
                        # Add detection info
                        cv2.rectangle(vis_img, (5, 5), (min(500, frame.shape[1]-5), 70), (0, 0, 0), -1)
                        cv2.putText(vis_img, f"Method: {method_name}", (10, 25), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.putText(vis_img, f"QR: {qr_data[:40]}...", (10, 45), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                        cv2.putText(vis_img, f"Time: {datetime.now().strftime('%H:%M:%S')}", (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    return qr_data, vis_img, processed_img
                    
            except Exception:
                continue
        
        return None, frame, None
    
    def log_detection(self, qr_content, method_name="Unknown"):
        """Log detected QR code with additional metadata, only if it's unique"""
        # Only log if this QR code hasn't been seen before
        if qr_content not in self.unique_qr_codes:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            detection = {
                'timestamp': timestamp,
                'qr_content': qr_content,
                'detection_method': method_name,
                'camera_type': 'PiCamera2' if self.picam2 else 'OpenCV',
                'resolution': f"{self.width}x{self.height}",
                'first_detection': True
            }
            self.detected_qr_codes.append(detection)
            self.unique_qr_codes.add(qr_content)
            print(f"🆕 New QR Code Detected: {qr_content}")
        else:
            print(f"🔄 Repeated QR Code: {qr_content} (already logged)")
    
    def save_detections(self):
        """Save detections to JSON with additional metadata"""
        if not self.detected_qr_codes:
            print("📝 No detections to save")
            return
            
        # Create output data structure with focus on unique detections
        output_data = {
            'metadata': {
                'device': 'Raspberry Pi 5',
                'camera_type': 'PiCamera2' if self.picam2 else 'OpenCV',
                'resolution': f"{self.width}x{self.height}",
                'unique_qr_codes': len(self.unique_qr_codes),
                'session_start': self.detected_qr_codes[0]['timestamp'],
                'session_end': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            'unique_detections': self.detected_qr_codes,  # Only contains first detection of each QR code
            'summary': {
                'total_unique_codes': len(self.unique_qr_codes),
                'detection_methods': list(set(d['detection_method'] for d in self.detected_qr_codes)),
                'qr_contents': list(self.unique_qr_codes)
            }
        }
        
        # Save to JSON file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"qr_detections_pi5_{timestamp}.json"
        filepath = os.path.join(OUTPUT_PATH, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved {len(self.detected_qr_codes)} detections to {filename}")
        
        # Also save a summary file
        summary_filename = f"qr_summary_pi5_{timestamp}.json"
        summary_filepath = os.path.join(OUTPUT_PATH, summary_filename)
        
        summary_data = {
            'session_info': output_data['metadata'],
            'unique_codes': list(set(d['qr_content'] for d in self.detected_qr_codes)),
            'detection_counts': {
                qr: len([d for d in self.detected_qr_codes if d['qr_content'] == qr])
                for qr in set(d['qr_content'] for d in self.detected_qr_codes)
            }
        }
        
        with open(summary_filepath, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Summary saved to {summary_filename}")
    
    def run_detection(self):
        """Main detection loop optimized for Pi 5"""
        print("🎥 Starting QR Live Detection on Raspberry Pi 5...")
        print("📋 Controls:")
        print("   'q' - Quit")
        print("   's' - Save detections")
        print("   'c' - Clear log")
        print("   'f' - Toggle full screen")
        print("="*50)
        
        last_qr_content = ""
        last_detection_time = 0
        frame_count = 0
        fps_start = time.time()
        fullscreen = False
        
        try:
            while True:
                ret, frame = self.capture_frame()
                if not ret or frame is None:
                    print("❌ Failed to capture frame")
                    time.sleep(0.1)
                    continue
                
                # Process for QR detection
                qr_content, vis_frame, _ = self.extract_qr_content_optimized(frame)
                
                # Handle QR detection (now using unique_qr_codes set)
                if qr_content:
                    current_time = time.time()
                    # Always log detection (the log_detection method handles uniqueness)
                    self.log_detection(qr_content)
                    last_qr_content = qr_content
                    last_detection_time = current_time
                
                # Add status overlay
                self.add_status_overlay(vis_frame, frame_count, fps_start)
                
                # Display frame
                window_name = 'QR Live Detection - Raspberry Pi 5'
                cv2.imshow(window_name, vis_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_detections()
                elif key == ord('c'):
                    self.detected_qr_codes.clear()
                    print("🔄 Detection log cleared")
                elif key == ord('f'):
                    fullscreen = not fullscreen
                    if fullscreen:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    else:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\n🛑 Detection stopped by user")
        finally:
            self.cleanup()
    
    def add_status_overlay(self, frame, frame_count, fps_start):
        """Add status information to frame"""
        # Calculate FPS every 30 frames
        if frame_count > 0 and frame_count % 30 == 0:
            fps = 30 / (time.time() - fps_start)
            fps_start = time.time()
        else:
            fps = 0
        
        # Status background
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (5, h-100), (300, h-5), (0, 0, 0), -1)
        
        # Status text
        cv2.putText(frame, f"Unique QR Codes: {len(self.unique_qr_codes)}", 
                   (10, h-80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Total Detections: {len(self.detected_qr_codes)}", 
                   (10, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if fps > 0:
            cv2.putText(frame, f"FPS: {fps:.1f}", 
                       (10, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Press 'q' to quit", 
                   (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def cleanup(self):
        """Clean up resources"""
        if self.picam2:
            self.picam2.stop()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Auto-save on exit
        if self.detected_qr_codes:
            self.save_detections()
            print(f"📊 Session completed with {len(self.detected_qr_codes)} total detections")

def main():
    """Run QR detection on Raspberry Pi 5"""
    print("🔧 Initializing QR Live Detection for Raspberry Pi 5...")
    
    # Create detector (will auto-select best camera method)
    detector = QRLiveDetectorPi5(width=1280, height=720, use_picamera2=True)
    
    # Start detection
    detector.run_detection()

if __name__ == "__main__":
    main()