# video_processor.py
"""
Video Processing Module for QR Code Scanner
Contains the VideoProcessor class responsible for video capture, QR detection, and visualization.
"""

import threading
import time
from datetime import datetime
from typing import Dict, Optional, Tuple
import logging
from queue import Queue, Empty, Full
import cv2
import os

from config.settings import QRScannerConfig
from core.qr_scanner import QRScanner
from api.json_manager_api import JsonLogManagerApi as JsonLogManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoProcessorApi:
    """
    API version of VideoProcessor for QR code scanning including capture, detection, and logging.
    Manages the main processing loop and coordinates between video capture and QR detection.
    """
    
    def __init__(self, config: QRScannerConfig):
        """Initialize video processor with configuration."""
        self.config = config
        self.qr_scanner = QRScanner(config)
        self.json_log_manager = JsonLogManager(config)
        
        # Video processing state
        self.video_source = None
        self.is_running = False
        self.processing_thread = None
        self.grabber_thread = None
        
        # Frame management
        self.frame_queue = Queue(maxsize=120)  # Increased buffer for frames to prevent blocking
        self.latest_frame = None
        
        # Performance metrics
        self.processing_fps = 0
        self.detection_time_ms = 0
        self.frames_processed = 0
        self.qr_codes_detected = 0
        
        # RTSP optimizations
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        
        logger.info("🎥 Video Processor initialized for QR code scanning")

    def start_processing(self, source: str) -> bool:
        """
        Start video processing in a separate thread.
        
        Args:
            source: Video source (camera index, file path, or RTSP URL)
            
        Returns:
            bool: True if processing started successfully, False otherwise.
        """
        logger.info(f"🚀 Starting video processing for source: {source}")
        
        # Initialize the video source
        if not self._initialize_video_source(source):
            logger.error("❌ Failed to initialize video source")
            return False

        self.is_running = True
        self.frames_processed = 0
        self.qr_codes_detected = 0

        # Start the frame grabber thread
        self.grabber_thread = threading.Thread(
            target=self._frame_grabber_loop,
            daemon=True
        )
        self.grabber_thread.start()
        logger.info("📹 Frame grabber thread started")

        # Start processing in daemon thread
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True
        )
        self.processing_thread.start()
        logger.info("🔍 Processing thread started")
        
        return True

    def stop_processing(self) -> Dict:
        """
        Stop video processing and return session summary.
        
        Returns:
            Dict: Session summary with statistics and detected QR codes
        """
        logger.info("🛑 Stopping video processing...")
        self.is_running = False
        
        # Wait for threads to finish
        if self.grabber_thread and self.grabber_thread.is_alive():
            self.grabber_thread.join(timeout=3.0)
            logger.info("📹 Frame grabber thread stopped")
            
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=3.0)
            logger.info("🔍 Processing thread stopped")
            
        # Clean up video source
        if self.video_source:
            self.video_source.release()
            self.video_source = None
            logger.info("🎥 Video source released")
        
        # Finalize logging
        self.json_log_manager.finalize_log()
        
        # Empty the queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break

        return self._create_session_summary()

    def _initialize_video_source(self, source: str) -> bool:
        """Initialize the video source with retry logic."""
        logger.info(f"🔗 Initializing video source: {source}")
        
        # Check if source is RTSP/HTTP stream
        is_rtsp = isinstance(source, str) and source.startswith(('rtsp://', 'http://', 'https://'))
        
        # Store source for later reference
        self.source = source
        
        max_retries = 8  # Increased retries for network stream stability
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                logger.info(f"🔄 Connection attempt {attempt + 1}/{max_retries} to {source}")
                
                # Use appropriate backend for RTSP streams
                if is_rtsp:
                    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                else:
                    cap = cv2.VideoCapture(source)
                
                if cap and cap.isOpened():
                    # Apply RTSP optimizations
                    if is_rtsp:
                        # Set a small buffer size for RTSP streams to prevent frame drops
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
                    
                    # Wait for stream to stabilize
                    time.sleep(2.0)
                    
                    # Verify we can actually read from the stream
                    logger.info(f"Attempt {attempt + 1}: Source connected. Verifying stream by grabbing a frame...")
                    grab_success = cap.grab()  # Try to grab one frame
                    if grab_success:
                        logger.info(f"Attempt {attempt + 1}: Test frame grabbed successfully. Stream is live.")
                        self.video_source = cap
                        logger.info("===================================================")
                        logger.info("   ✅ SYSTEM READY: QR Scanner can now proceed.   ")
                        logger.info("===================================================")
                        return True
                    else:
                        logger.warning(f"Attempt {attempt + 1}: Connected but failed to grab frame. Stream may not be ready.")
                        cap.release()  # Release the faulty connection
                else:
                    logger.warning(f"Attempt {attempt + 1}: Failed to open source.")
                    if cap:
                        cap.release()
                
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                
            except Exception as e:
                logger.error(f"❌ Connection error (attempt {attempt + 1}): {e}")
                time.sleep(retry_delay)
        
        logger.error(f"❌ Failed to connect after {max_retries} attempts")
        return False

    def _read_frame(self) -> Tuple[bool, Optional[cv2.Mat]]:
        """Read a single frame from the video source."""
        if not self.video_source or not self.video_source.isOpened():
            return False, None
        
        try:
            # For RTSP streams, grab then retrieve is more stable
            ret = self.video_source.grab()
            if not ret:
                return False, None
            
            ret, frame = self.video_source.retrieve()
            if not ret or frame is None:
                return False, None
            
            # Validate frame
            if frame.size == 0:
                return False, None
                
            return True, frame
            
        except Exception as e:
            logger.warning(f"⚠️ Error reading frame: {e}")
            return False, None

    def _frame_grabber_loop(self):
        """
        Dedicated loop for grabbing frames from the source. It uses a blocking
        'put' to ensure that every frame is queued for processing without being
        dropped, waiting if the processing thread falls behind.
        """
        logger.info("📹 Frame grabber loop started")
        frames_grabbed = 0
        
        while self.is_running:
            ret, frame = self._read_frame()
            if not ret:
                logger.warning(f"Failed to grab frame after {frames_grabbed} frames. Stopping grabber.")
                # Place a sentinel value to signal the end to the processor
                try:
                    self.frame_queue.put(None, timeout=0.5)
                except Full:
                    logger.warning("Frame queue was full when trying to add sentinel. Processing may be stuck.")
                break

            # This is a blocking call. If the queue is full, this thread will
            # wait until a slot is available. This is the key to preventing
            # frame drops and ensuring every frame is processed.
            try:
                self.frame_queue.put((frame, datetime.now(self.config.TIMEZONE)), timeout=1.0)
                frames_grabbed += 1
                
                # Log progress occasionally
                if frames_grabbed % 30 == 0:
                    logger.info(f"📹 Grabbed {frames_grabbed} frames successfully")
                    
            except Full:
                logger.warning("Frame queue is full. Frame grabber is blocked, indicating processing is too slow.")
                # If the queue is full, we wait. If we need to stop, is_running will be false.
                continue

        logger.info(f"📹 Frame grabber loop finished after grabbing {frames_grabbed} frames")

    def _processing_loop(self):
        """Main processing loop for QR code detection."""
        logger.info("🔍 Processing loop started")
        print("🚀 [THREAD] QR Processing Thread Started - Ready for QR Detection!")
        frame_idx = 0
        fps_start_time = time.time()
        fps_frame_count = 0
        
        while self.is_running:
            try:
                # Get frame from queue, waiting up to 1 second
                item = self.frame_queue.get(timeout=1.0)
                if item is None:  # Sentinel value means grabber stopped
                    break
                    
                frame, timestamp = item
                frame_idx += 1
                self.frames_processed += 1
                
                # Process every frame for QR detection
                detection_start_time = time.time()
                processed_frame, qr_data = self.qr_scanner.process_frame(frame)
                self.detection_time_ms = (time.time() - detection_start_time) * 1000
                
                # Handle QR code detection
                if qr_data:
                    self.qr_codes_detected += 1
                    self.json_log_manager.add_record(qr_data)
                    logger.info(f"✅ QR Code detected: {qr_data['content']}")
                    
                    # Print a clear console message for QR detection
                    print(f"\n🎯 [QR DETECTION] New QR Code Found!")
                    print(f"📋 Content: {qr_data['content']}")
                    print(f"⏰ Timestamp: {qr_data['timestamp']}")
                    print(f"📊 Total Detected: {self.qr_codes_detected}")
                    print(f"🎥 Processing FPS: {self.processing_fps:.2f}")
                    print("─" * 50)
                
                # Calculate FPS
                fps_frame_count += 1
                if time.time() - fps_start_time >= 1.0:
                    self.processing_fps = fps_frame_count / (time.time() - fps_start_time)
                    fps_frame_count = 0
                    fps_start_time = time.time()
                
                # Store latest frame for display
                self.latest_frame = processed_frame.copy()
                
            except Empty:
                logger.warning("Frame queue was empty for 1 second. Assuming stream ended.")
                break  # Exit if the queue is empty for too long
            except Exception as e:
                logger.error(f"❌ Error in processing loop: {e}")
                time.sleep(0.1)
        
        self.is_running = False
        logger.info("🔍 Processing loop finished")
        print("🛑 [THREAD] QR Processing Thread Stopped")

    def get_status(self) -> Dict:
        """Get current processing status."""
        status = {
            "is_running": self.is_running,
            "processing_fps": self.processing_fps,
            "detection_time_ms": self.detection_time_ms,
            "frames_processed": self.frames_processed,
            "qr_codes_detected": self.qr_codes_detected,
            "queue_size": self.frame_queue.qsize(),
            "has_latest_frame": self.latest_frame is not None
        }
        
        # Add stream information if available
        stream_info = self.get_stream_info()
        if stream_info:
            status["stream_info"] = stream_info
        
        return status

    def get_latest_frame(self) -> Optional[cv2.Mat]:
        """Get the latest processed frame."""
        return self.latest_frame
    
    def get_stream_info(self) -> Optional[Dict]:
        """Get information about the current video stream."""
        if not self.video_source or not self.video_source.isOpened():
            return None
        
        try:
            width = int(self.video_source.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.video_source.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.video_source.get(cv2.CAP_PROP_FPS)
            buffer_size = self.video_source.get(cv2.CAP_PROP_BUFFERSIZE)
            
            return {
                "width": width,
                "height": height,
                "fps": fps,
                "buffer_size": buffer_size,
                "source": self.source,
                "is_rtsp": isinstance(self.source, str) and self.source.startswith(('rtsp://', 'http://', 'https://'))
            }
        except Exception as e:
            logger.error(f"❌ Error getting stream info: {e}")
            return None

    def _create_session_summary(self) -> Dict:
        """Create session summary with statistics."""
        return {
            "total_frames_processed": self.frames_processed,
            "total_qr_codes_detected": self.qr_codes_detected,
            "average_fps": self.processing_fps,
            "average_detection_time_ms": self.detection_time_ms,
            "log_file": self.json_log_manager.output_file,
            "session_duration": "N/A"  # Could be calculated if needed
        }

    def __del__(self):
        """Cleanup on destruction."""
        if self.is_running:
            self.stop_processing()
