"""
Flask API for QR Code Scanner
Provides remote control endpoints for starting and stopping QR code scanning.

Supported video sources:
- RTSP/HTTP streams (e.g., "rtsp://localhost:8554/cam1")
- Camera indices (e.g., "0", "1")
- File paths (e.g., "/path/to/video.mp4")
- PiCamera (use "picamera" as video_source)

Example usage:
POST /start
{
    "video_source": "picamera"  # or any other supported source
}
"""

import json
import threading
import time
import os
from datetime import datetime
from typing import Dict, Optional, List
import cv2

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from config.settings import QRScannerConfig
from api.video_processor_api import VideoProcessorApi as VideoProcessor

# Using RTSPVideoStreamer directly - no need for EnhancedVideoStreamer
from core.qr_scanner import QRScanner
from api.json_manager_api import JsonLogManagerApi as JsonLogManager

# PiCamera support
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
        # Note: tuning_file should be in the same directory as this script
        tuning_file = "imx219_noir.json"
        self.picam2 = Picamera2()

        try:
            self.picam2.load_tuning_file(tuning_file)
        except Exception as e:
            print(f"⚠️ Warning: Could not load tuning file {tuning_file}: {e}")
            print("   Using default camera configuration")
        config = self.picam2.create_video_configuration(
                    main={"size": (width, height), "format": "RGB888"},
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
        try:
            if hasattr(self, 'picam2') and self.picam2:
                self.picam2.stop()
                self.picam2.close()
                print("📷 PiCamera stopped and closed successfully")
        except Exception as e:
            print(f"⚠️ Error stopping PiCamera: {e}")
        finally:
            # Force cleanup
            try:
                import gc
                gc.collect()
            except:
                pass
        
    def connect(self):
        """Dummy connect method to match VideoStreamer interface."""
        return True
        
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            if hasattr(self, 'picam2') and self.picam2:
                self.picam2.stop()
                self.picam2.close()
                print("📷 PiCamera cleaned up in destructor")
        except:
            pass

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables
qr_scanner: Optional[QRScanner] = None
# video_streamer: Optional[VideoStreamer] = None  # Not used anymore
json_log_manager: Optional[JsonLogManager] = None
scanning_thread: Optional[threading.Thread] = None

# Control variables
is_scanning = False
should_stop = False

# Status tracking
scanning_status = {
    "is_running": False,
    "start_time": None,
    "current_session": None,
    "detected_codes": [],
    "fps": 0.0
}

class QRScanningThread:
    """Thread class for running QR scanning in background"""
    
    def __init__(self, video_source: str, config: QRScannerConfig):
        self.video_source = video_source
        self.config = config
        
        # Determine the appropriate video source
        self.video_source = self._determine_video_source(video_source)
        
        # Use VideoProcessor instead of separate components
        self.video_processor = VideoProcessor(config)
        self.is_running = False
        
        # PiCamera statistics
        self.picamera_stats = {
            "total_qr_codes_detected": 0,
            "total_frames_processed": 0,
            "average_fps": 0.0,
            "average_detection_time_ms": 0.0,
            "log_file": ""
        }
        
    def _determine_video_source(self, source) -> str:
        """Determine the appropriate video source based on input"""
        # Convert source to string if it's not already
        source_str = str(source)
        
        # If it's picamera, return as is
        if source_str == "picamera":
            return source_str
            
        # If it's a number, treat as camera index
        if source_str.isdigit():
            return int(source_str)
        
        # If it's a file path, return as is
        if os.path.exists(source_str):
            return source_str
            
        # If it's an RTSP URL, return as is
        if source_str.startswith(('rtsp://', 'http://', 'https://')):
            return source_str
            
        # If it's a webcam device path (Linux)
        if source_str.startswith('/dev/video'):
            return source_str
            
        # Default to camera 0
        print(f"⚠️ Unknown video source format: {source_str}. Using camera 0.")
        return 0
        
    def start(self):
        """Start the scanning thread"""
        if self.is_running:
            return False
            
        # Handle picamera differently
        if self.video_source == "picamera":
            try:
                # Create PiCamera streamer with config settings
                self.video_processor = PiCameraStreamer(
                    width=self.config.CAMERA_RESOLUTION[0] if hasattr(self.config, 'CAMERA_RESOLUTION') else 1280,
                    height=self.config.CAMERA_RESOLUTION[1] if hasattr(self.config, 'CAMERA_RESOLUTION') else 960,
                    framerate=self.config.FRAME_RATE if hasattr(self.config, 'FRAME_RATE') else 15
                )
                # Start the processing loop for PiCamera
                self.processing_thread = threading.Thread(target=self._picamera_processing_loop, daemon=True)
                self.processing_thread.start()
                self.is_running = True
                print("📷 PiCamera processing thread started")
                return True
            except Exception as e:
                print(f"❌ Failed to initialize PiCamera: {str(e)}")
                return False
        else:
            # Start video processing for other sources
            if not self.video_processor.start_processing(self.video_source):
                return False
            
        self.is_running = True
        return True
        

        
    def stop(self):
        """Stop the scanning thread"""
        self.is_running = False
        
        if self.video_source == "picamera":
            try:
                # Stop the processing thread
                if hasattr(self, 'processing_thread') and self.processing_thread and self.processing_thread.is_alive():
                    self.processing_thread.join(timeout=3.0)
                    print("📷 PiCamera processing thread stopped")
                
                # Stop the camera
                if hasattr(self.video_processor, 'stop'):
                    self.video_processor.stop()
                    print("📷 PiCamera processor stopped successfully")
            except Exception as e:
                print(f"⚠️ Error stopping PiCamera processor: {e}")
            
            # Force cleanup for PiCamera
            try:
                import gc
                gc.collect()
                time.sleep(1)  # Give time for cleanup
            except:
                pass
                
            # Return actual session summary for picamera
            return self.picamera_stats
        else:
            return self.video_processor.stop_processing()
        
    def get_status(self):
        """Get current status from video processor"""
        if self.video_processor:
            if self.video_source == "picamera":
                # For picamera, return basic status
                return {
                    "source": "picamera",
                    "is_running": self.is_running,
                    "status": "active" if self.is_running else "stopped"
                }
            else:
                return self.video_processor.get_status()
        return None
        
    def _picamera_processing_loop(self):
        """Main processing loop for PiCamera QR detection."""
        print("🔍 PiCamera processing loop started")
        
        # Import QRScanner here to avoid circular imports
        from core.qr_scanner import QRScanner
        from api.json_manager_api import JsonLogManagerApi as JsonLogManager
        
        qr_scanner = QRScanner(self.config)
        json_log_manager = JsonLogManager(self.config)
        
        frame_count = 0
        fps_start_time = time.time()
        fps_frame_count = 0
        qr_codes_detected = 0
        total_detection_time_ms = 0
        
        while self.is_running:
            try:
                # Capture frame from PiCamera
                ret, frame = self.video_processor.read()
                
                if not ret or frame is None:
                    print("⚠️ PiCamera: Could not read frame")
                    time.sleep(0.1)
                    continue
                
                frame_count += 1
                fps_frame_count += 1
                
                # Process frame for QR detection
                detection_start_time = time.time()
                processed_frame, qr_data = qr_scanner.process_frame(frame)
                detection_time_ms = (time.time() - detection_start_time) * 1000
                total_detection_time_ms += detection_time_ms
                
                # Handle QR code detection
                if qr_data:
                    qr_codes_detected += 1
                    json_log_manager.add_record(qr_data)
                    print(f"🎯 [PiCamera] QR Code detected: {qr_data['content']}")
                    
                    # Update global scanning status
                    global scanning_status
                    if 'detected_codes' not in scanning_status:
                        scanning_status['detected_codes'] = []
                    scanning_status['detected_codes'].append(qr_data)
                
                # Calculate FPS
                if time.time() - fps_start_time >= 1.0:
                    fps = fps_frame_count / (time.time() - fps_start_time)
                    scanning_status['fps'] = round(fps, 2)
                    fps_frame_count = 0
                    fps_start_time = time.time()
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                print(f"❌ Error in PiCamera processing loop: {e}")
                time.sleep(0.1)
        
        print("🔍 PiCamera processing loop finished")
        
        # Finalize logging first
        try:
            json_log_manager.finalize_log()
        except Exception as e:
            print(f"⚠️ Error finalizing log: {e}")
        
        # Update statistics
        self.picamera_stats["total_qr_codes_detected"] = qr_codes_detected
        self.picamera_stats["total_frames_processed"] = frame_count
        self.picamera_stats["average_fps"] = round(frame_count / (time.time() - fps_start_time + 1), 2) if frame_count > 0 else 0.0
        self.picamera_stats["average_detection_time_ms"] = round(total_detection_time_ms / frame_count, 2) if frame_count > 0 else 0.0
        self.picamera_stats["log_file"] = json_log_manager.output_file if hasattr(json_log_manager, 'output_file') and json_log_manager.output_file else ""

# Global scanning thread instance
scanning_thread_instance: Optional[QRScanningThread] = None

def _get_log_content(log_file_path: str) -> dict:
    """
    Read and return the content of the log file.
    
    Args:
        log_file_path: Path to the log file
        
    Returns:
        dict: Content of the log file as JSON, or error message if file cannot be read
    """
    if not log_file_path:
        return {"error": "Log file path is empty"}
    
    if not os.path.exists(log_file_path):
        print(f"⚠️ Log file not found: {log_file_path}")
        return {"error": f"Log file not found: {log_file_path}"}
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
        print(f"✅ Log file read successfully: {log_file_path}")
        return content
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in log file: {log_file_path} - {str(e)}")
        return {"error": f"Invalid JSON in log file: {str(e)}"}
    except Exception as e:
        print(f"❌ Error reading log file: {log_file_path} - {str(e)}")
        return {"error": f"Error reading log file: {str(e)}"}

@app.route('/start', methods=['POST'])
def start_scanning():
    """
    Start QR code scanning with provided parameters.
    
    Expected JSON payload:
    {
        "video_source": "rtsp://127.0.0.1:8554/cam1"  # or camera URL/file path, or "picamera"
    }
    """
    global scanning_thread_instance, scanning_status
    
    try:
        # Check if already running
        if scanning_status["is_running"]:
            return jsonify({
                "success": False,
                "message": "QR scanning is already running",
                "status": scanning_status
            }), 400
        
        # Get parameters from request - handle both JSON and form data
        data = {}
        if request.is_json:
            data = request.get_json() or {}
        else:
            # Handle form data or empty request
            data = request.form.to_dict() or {}
        
        # Try different video sources in order of preference
        video_source = data.get("video_source", "rtsp://localhost:8554/cam1")
        
        # Check if picamera is requested
        if video_source == "picamera":
            if not PICAMERA2_AVAILABLE:
                return jsonify({
                    "success": False,
                    "message": "PiCamera2 library is not available. Please install with: pip install picamera2",
                    "status": scanning_status
                }), 400
            print("📷 PiCamera requested as video source")
        # Check if video_source is an RTSP/HTTP stream
        elif isinstance(video_source, str) and video_source.startswith(('rtsp://', 'http://', 'https://')):
            print(f"🌐 RTSP/HTTP stream detected: {video_source}")
            # For RTSP streams, we don't need to search for cameras
            pass
        elif not video_source:
            # If no specific source provided or it's "0", try to find a working camera
            print("🔍 No specific video source provided, searching for available cameras...")
            working_camera_found = False
            
            # Try to find a working camera with different backends
            for i in range(5):  # Try cameras 0-4
                for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
                    try:
                        print(f"🔍 Testing camera {i} with backend {backend}")
                        test_cap = cv2.VideoCapture(i, backend)
                        if test_cap and test_cap.isOpened():
                            ret, frame = test_cap.read()
                            test_cap.release()
                            if ret and frame is not None:
                                video_source = str(i)
                                print(f"✅ Found working camera: {i} with backend {backend}")
                                working_camera_found = True
                                break
                    except Exception as e:
                        print(f"⚠️ Camera {i} with backend {backend} failed: {e}")
                        continue
                
                if working_camera_found:
                    break
            
            if not working_camera_found:
                return jsonify({
                    "success": False,
                    "message": "No working camera found. Please provide an RTSP URL or check your camera connection.",
                    "status": scanning_status
                }), 500
        
        print(f"🎥 Starting QR scanning with video source: {video_source}")
        
        # Create configuration
        config = QRScannerConfig()
        
        # Create and start scanning thread
        scanning_thread_instance = QRScanningThread(video_source, config)
        
        if not scanning_thread_instance.start():
            return jsonify({
                "success": False,
                "message": "Failed to start QR scanning. Please check the video source.",
                "status": scanning_status
            }), 500
        
        # Update status
        scanning_status["is_running"] = True
        scanning_status["start_time"] = datetime.now(config.TIMEZONE).strftime('%Y-%m-%d %H:%M:%S')
        scanning_status["current_session"] = {
            "video_source": video_source
        }
        scanning_status["detected_codes"] = []
        scanning_status["fps"] = 0.0
        
        return jsonify({
            "success": True,
            "message": "QR scanning started successfully",
            "status": scanning_status
        }), 200
        
    except Exception as e:
        print(f"❌ Error in start_scanning: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Failed to start scanning: {str(e)}",
            "status": scanning_status
        }), 500

@app.route('/stop', methods=['POST'])
def stop_scanning():
    """
    Stop QR code scanning and return session summary.
    """
    global scanning_thread_instance, scanning_status
    
    try:
        print("🛑 Stopping QR scanning...")
        
        # Check if not running
        if not scanning_status["is_running"] or scanning_thread_instance is None:
            return jsonify({
                "success": False,
                "message": "No scanning session is currently running",
                "status": scanning_status
            }), 400
        
        # Stop scanning and get session summary
        session_summary = scanning_thread_instance.stop()
        
        # Create summary using session data with safe access
        summary = {
            "total_codes_detected": session_summary.get("total_qr_codes_detected", 0) if session_summary else 0,
            "start_time": scanning_status["start_time"],
            "end_time": datetime.now(QRScannerConfig().TIMEZONE).strftime('%Y-%m-%d %H:%M:%S'),
            "video_source": scanning_status["current_session"].get("video_source", "unknown") if scanning_status["current_session"] else "unknown",
            "detected_codes": scanning_status["detected_codes"].copy(),  # Copy before clearing
            "log_file": session_summary.get("log_file", "") if session_summary else "",
            "log_content": _get_log_content(session_summary.get("log_file", "")) if session_summary else "",
            "processing_stats": {
                "total_frames_processed": session_summary.get("total_frames_processed", 0) if session_summary else 0,
                "average_fps": session_summary.get("average_fps", 0.0) if session_summary else 0.0,
                "average_detection_time_ms": session_summary.get("average_detection_time_ms", 0.0) if session_summary else 0.0
            }
        }
        
        # Update status
        scanning_status["is_running"] = False
        scanning_status["start_time"] = None
        scanning_status["current_session"] = None
        scanning_status["detected_codes"] = []
        scanning_status["fps"] = 0.0
        
        # Clean up
        scanning_thread_instance = None
        
        print("✅ QR scanning stopped successfully")
        
        return jsonify({
            "success": True,
            "message": "QR scanning stopped successfully",
            "summary": summary,
            "status": scanning_status
        }), 200
        
    except Exception as e:
        print(f"❌ Error in stop_scanning: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Failed to stop scanning: {str(e)}",
            "status": scanning_status
        }), 500

@app.route('/status', methods=['GET'])
def get_status():
    """
    Get current scanning status and statistics.
    """
    global scanning_status, scanning_thread_instance
    
    try:
        status_info = {
            "is_running": scanning_status["is_running"],
            "start_time": scanning_status["start_time"],
            "current_session": scanning_status["current_session"],
            "fps": scanning_status["fps"],
            "total_codes_detected": len(scanning_status["detected_codes"]),
            "latest_codes": scanning_status["detected_codes"][-10:] if scanning_status["detected_codes"] else []  # Last 10 codes
        }
        
        # Add processing information if available
        if scanning_thread_instance and scanning_thread_instance.is_running:
            processor_status = scanning_thread_instance.get_status()
            status_info["processor_status"] = processor_status
        
        return jsonify({
            "success": True,
            "status": status_info
        }), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Failed to get status: {str(e)}",
            "status": scanning_status
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Simple health check endpoint.
    """
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now(QRScannerConfig().TIMEZONE).strftime('%Y-%m-%d %H:%M:%S'),
        "service": "QR Code Scanner API"
    }), 200

@app.route('/test', methods=['GET', 'POST'])
def test_endpoint():
    """
    Test endpoint to verify API is working.
    """
    return jsonify({
        "success": True,
        "message": "QR Scanner API is working!",
        "method": request.method,
        "timestamp": datetime.now(QRScannerConfig().TIMEZONE).strftime('%Y-%m-%d %H:%M:%S')
    }), 200









if __name__ == '__main__':
    print("🚀 Starting QR Code Scanner Flask API with Video Processor...")
    print("📡 Available endpoints:")
    print("   POST /start - Start QR scanning (default: rtsp://localhost:8554/cam1)")
    print("   POST /stop  - Stop QR scanning")
    print("   GET  /status - Get current status with processing info")
    print("   GET  /health - Health check")


    print("\n🌐 Server will start on http://localhost:5002")
    print("💡 Default RTSP stream: rtsp://localhost:8554/cam1")
    print("💡 To start scanning: curl -X POST http://localhost:5002/start")
    
    app.run(host='0.0.0.0', port=5002, debug=False) 