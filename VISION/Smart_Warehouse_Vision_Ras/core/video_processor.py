# core/video_processor.py
"""
Video Processing Module
Contains the VideoProcessor class responsible for video capture, model inference, and visualization.
"""

import threading
import time
from datetime import datetime
from typing import Dict
import logging
import torch
from queue import Queue, Empty, Full

import cv2
import streamlit as st
from ultralytics import YOLO

from config.warehouse_config import WarehouseConfig
from core.tracker import ProductTracker

# Get a logger for this module
logger = logging.getLogger(__name__)

# Conditionally import picamera2
try:
    from picamera2 import Picamera2
    IS_PICAMERA_AVAILABLE = True
    logger.info("picamera2 library found.")
except ImportError:
    IS_PICAMERA_AVAILABLE = False
    logger.info("picamera2 library not found, falling back to cv2.VideoCapture.")


class VideoProcessor:
    """
    Handles video processing pipeline including model inference, tracking integration, and visualization.
    Manages the main processing loop and coordinates between YOLO model and ProductTracker.
    """
    
    def __init__(self):
        """Initialize video processor with default state."""
        self.model = None
        self.latest_frame = None
        self.is_running = False
        self.tracker = ProductTracker()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.processing_thread = None
        self.grabber_thread = None # Thread for grabbing frames
        # Use a large queue to buffer frames, ensuring none are dropped.
        # The grabber will wait if this queue is full.
        self.frame_queue = Queue(maxsize=120)
        self.processing_fps = 0
        self.video_source = None # Will be cv2.VideoCapture or Picamera2 instance
        logger.info(f"Using device: {self.device}")
        logger.info(f"VideoProcessor initialized. IS_PICAMERA_AVAILABLE: {IS_PICAMERA_AVAILABLE}")

    def initialize_model(self, weights_path):
        """
        Initialize YOLO model with given weights.
        
        Args:
            weights_path: Path to YOLO weights file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.model = YOLO(weights_path)
            self.model.to(self.device)
            logger.info(f"Model '{weights_path}' loaded successfully onto {self.device}.")
            return True
        except Exception as e:
            st.error(f"Model load failed: {e}")
            logger.error(f"Failed to load model '{weights_path}': {e}", exc_info=True)
            return False

    def start_processing(self, source, weights_path, frame_skip, conf_thresh, location, model_input_size):
        """
        Start video processing in a separate thread.
        
        Args:
            source: Video source (camera index, file path, stream URL, or 'picamera')
            weights_path: Path to YOLO weights
            frame_skip: Number of frames to skip between processing
            conf_thresh: Confidence threshold for detection
            location: Location name for this processing session
            model_input_size: The input size for the model (e.g., 640, 1280)
            
        Returns:
            bool: True if processing started successfully, False otherwise.
        """
        if not self.initialize_model(weights_path):
            return False
            
        self.tracker.reset()
        
        # Initialize the video source. If it fails, stop here.
        if not self._initialize_video_source(source):
            self.is_running = False
            return False

        self.is_running = True

        # Start the frame grabber thread
        self.grabber_thread = threading.Thread(
            target=self._frame_grabber_loop,
            daemon=True
        )
        self.grabber_thread.start()

        # Start processing in daemon thread
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            args=(frame_skip, conf_thresh, location, model_input_size),
            daemon=True
        )
        self.processing_thread.start()
        return True

    def stop_processing(self):
        """
        Stop video processing and return session summary.
        
        Returns:
            Dict: Session summary with events and statistics
        """
        self.is_running = False
        
        # Wait for the processing and grabber threads to finish
        if self.grabber_thread and self.grabber_thread.is_alive():
            self.grabber_thread.join()
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join()
            
        # Clean up video source
        if self.video_source:
            if IS_PICAMERA_AVAILABLE and isinstance(self.video_source, Picamera2):
                if self.video_source.started:
                    self.video_source.stop()
            elif isinstance(self.video_source, cv2.VideoCapture):
                self.video_source.release()
            self.video_source = None
        
        # Empty the queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break

        events = list(self.tracker.events)
        counts = dict(self.tracker.counts)
        return self._create_session_summary(events, counts)

    def _initialize_video_source(self, source):
        """Initialize the correct video source based on input."""
        logger.info(f"Attempting to initialize video source: {source}")
        if source == "picamera" and IS_PICAMERA_AVAILABLE:
            try:
                logger.info("Initializing Raspberry Pi camera.")
                picam2 = Picamera2()
                config = picam2.create_video_configuration(main={"size": (1280, 720), "format": "RGB888"})
                picam2.configure(config)
                picam2.start()
                time.sleep(2.0) # Allow camera to warm up
                self.video_source = picam2
                logger.info("picamera2 started successfully.")
                return True
            except Exception as e:
                logger.error(f"Failed to initialize picamera2: {e}", exc_info=True)
                st.error(f"PiCamera Error: {e}. Please ensure it is connected and enabled.")
                return False
        elif source == "picamera":
             logger.error("Pi Camera was selected, but the picamera2 library is not available.")
             st.error("Pi Camera was selected, but the `picamera2` library is not installed.")
             return False
        else:
            try:
                # Set environment variables for OpenCV to improve RTSP stream handling.
                # Use TCP transport for reliability and increase probe size for better stream analysis.
                import os
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
                
                logger.info(f"Initializing cv2.VideoCapture for source: {source}")

                # Retry logic to handle intermittent connection issues where the stream may not be ready immediately.
                self.video_source = None
                max_retries = 8  # Increased retries for network stream stability
                retry_delay = 2  # seconds
                for attempt in range(max_retries):
                    logger.info(f"Connection attempt {attempt + 1}/{max_retries} to {source}")
                    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                    
                    if cap and cap.isOpened():
                        # The stream is open, now verify we can actually read from it.
                        logger.info(f"Attempt {attempt + 1}: Source connected. Verifying stream by grabbing a frame...")
                        grab_success = cap.grab()  # Try to grab one frame
                        if grab_success:
                            logger.info(f"Attempt {attempt + 1}: Test frame grabbed successfully. Stream is live.")
                            self.video_source = cap
                            break  # Success! Exit the loop.
                        else:
                            logger.warning(f"Attempt {attempt + 1}: Connected but failed to grab frame. Stream may not be ready.")
                            cap.release()  # Release the faulty connection
                    else:
                        logger.warning(f"Attempt {attempt + 1}: Failed to open source.")
                        if cap:
                            cap.release()

                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                
                if not self.video_source:
                    logger.error(f"Failed to open video source after {max_retries} attempts: {source}")
                    st.error(f"Cannot open video source: {source}. Please check URL and ensure stream is active.")
                    return False

                # Set a small buffer size. This can help stabilize some RTSP streams
                # by ensuring there's always a frame ready, preventing timeouts.
                self.video_source.set(cv2.CAP_PROP_BUFFERSIZE, 3)
                    
                logger.info(f"cv2.VideoCapture opened source: {source} successfully.")
                logger.info("===================================================")
                logger.info("   ✅ SYSTEM READY: Forklift can now proceed.   ")
                logger.info("===================================================")
                return True
            except Exception as e:
                logger.error(f"Failed to initialize cv2.VideoCapture for source {source}: {e}", exc_info=True)
                st.error(f"Failed to open video source: {source}")
                return False

    def _read_frame(self):
        """Read a single frame from the initialized video source."""
        if IS_PICAMERA_AVAILABLE and isinstance(self.video_source, Picamera2):
            return True, self.video_source.capture_array()
        
        if isinstance(self.video_source, cv2.VideoCapture):
            # For some backends, especially with RTSP, grabbing the frame first 
            # and then retrieving it can be more stable than a single read() call.
            ret = self.video_source.grab()
            if not ret:
                return False, None
            return self.video_source.retrieve()

        return False, None

    def _frame_grabber_loop(self):
        """
        Dedicated loop for grabbing frames from the source. It uses a blocking
        'put' to ensure that every frame is queued for processing without being
        dropped, waiting if the processing thread falls behind.
        """
        logger.info("Frame grabber loop started.")
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
                self.frame_queue.put((frame, datetime.now(WarehouseConfig.TIMEZONE)), timeout=1.0)
                frames_grabbed += 1
            except Full:
                logger.warning("Frame queue is full. Frame grabber is blocked, indicating processing is too slow.")
                # If the queue is full, we wait. If we need to stop, is_running will be false.
                continue

        logger.info(f"Frame grabber loop finished after grabbing {frames_grabbed} frames.")


    def _processing_loop(self, frame_skip, conf_thresh, location, model_input_size):
        """
        Main processing loop that runs in a separate thread.
        Handles video capture, frame processing, and visualization.
        """
        frame_idx = 0
        fps_start_time = time.time()
        fps_frame_count = 0

        while self.is_running:
            try:
                # Get a frame from the queue, waiting up to 1 second
                item = self.frame_queue.get(timeout=1.0)
                if item is None: # Sentinel value means grabber stopped
                    break
                frame, timestamp_obj = item
            except Empty:
                logger.warning("Frame queue was empty for 1 second. Assuming stream ended.")
                break # Exit if the queue is empty for too long

            frame_idx += 1
            
            # Only process frame if the skip interval is met
            if frame_idx % frame_skip == 0:
                # FPS calculation
                fps_frame_count += 1
                if time.time() - fps_start_time >= 1.0:
                    self.processing_fps = fps_frame_count / (time.time() - fps_start_time)
                    fps_frame_count = 0
                    fps_start_time = time.time()
                
                # Run YOLO inference
                results = self.model(frame, imgsz=model_input_size, verbose=False)[0]
                boxes = results.boxes.xyxy.cpu().numpy()
                class_ids = results.boxes.cls.cpu().numpy()
                confidences = results.boxes.conf.cpu().numpy()
                
                # Filter detections and format for tracker
                detections = [
                    (tuple(box.astype(int)), int(class_id), conf)
                    for box, class_id, conf in zip(boxes, class_ids, confidences)
                    if conf >= conf_thresh and int(class_id) in WarehouseConfig.VALID_CLASSES
                ]
                
                # Update tracker with new detections
                self.tracker.update_tracks(detections, frame, timestamp_obj, location)
            
            # Always draw visualization on the frame that was just processed.
            # This ensures the UI is perfectly synchronized with the tracker's state.
            self._draw_visualization(frame)
            
            # Store latest frame for display in Streamlit
            self.latest_frame = frame.copy()

            # Avoid busy-waiting on skipped frames
            if frame_idx % frame_skip != 0:
                 time.sleep(0.01)
            
        self.is_running = False
        logger.info("Processing loop finished.")

    def _draw_visualization(self, frame):
        """
        Draw bounding boxes, track IDs, and the counting zone on the frame.
        
        Args:
            frame: OpenCV frame to draw on
        """
        # Draw the counting zone
        frame_h, frame_w, _ = frame.shape
        zone_x1 = int(frame_w * WarehouseConfig.COUNTING_ZONE_X_START_RATIO)
        zone_x2 = int(frame_w * WarehouseConfig.COUNTING_ZONE_X_END_RATIO)
        zone_y1 = int(frame_h * WarehouseConfig.COUNTING_ZONE_Y_START_RATIO)
        zone_y2 = int(frame_h * WarehouseConfig.COUNTING_ZONE_Y_END_RATIO)
        
        # Draw the zone with a semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (zone_x1, zone_y1), (zone_x2, zone_y2), (255, 100, 0), -1) # Blue, filled
        alpha = 0.2
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw the zone border
        cv2.rectangle(frame, (zone_x1, zone_y1), (zone_x2, zone_y2), (255, 150, 50), 2) # Brighter blue border

        for track_id, track in self.tracker.tracks.items():
            x1, y1, x2, y2 = track.bbox
            
            # Determine color based on track state
            if track.state == 'CONFIRMED':
                color = (0, 255, 0)      # Green
            elif track.state == 'COASTING':
                color = (255, 165, 0)    # Orange
            elif track.state == 'TENTATIVE':
                color = (0, 255, 255)    # Yellow
            else: # COUNTED or other states
                color = (255, 0, 0)      # Blue

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Add track ID and state text
            label = f"ID:{track_id} [{track.state}]"
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Display counts
        counts_text = f"Loaded: {self.tracker.counts['in']} | Unloaded: {self.tracker.counts['out']}"
        cv2.putText(frame, counts_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Display processing FPS
        fps_text = f"FPS: {self.processing_fps:.1f}"
        cv2.putText(frame, fps_text, (frame.shape[1] - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    def _create_session_summary(self, events, counts):
        """
        Create session summary from events and counts.
        
        Args:
            events: List of tracking events
            counts: Dictionary of in/out counts
            
        Returns:
            Dict: Formatted session summary
        """
        loaded_count = counts.get('in', 0)
        unloaded_count = counts.get('out', 0)
        
        detailed_product_counts = {"loaded": {}, "unloaded": {}}
        if events:
            for event_data in events:
                status = event_data[1]
                product_type = event_data[4]
                if status in detailed_product_counts:
                    detailed_product_counts[status][product_type] = detailed_product_counts[status].get(product_type, 0) + 1
        
        if not events and loaded_count == 0 and unloaded_count == 0:
            operation_type = "none"
        elif loaded_count > 0 and unloaded_count == 0:
            operation_type = "loading"
        elif unloaded_count > 0 and loaded_count == 0:
            operation_type = "unloading"
        else:
            operation_type = "mixed"
            
        start_time = events[0][0] if events else "N/A"
        end_time = events[-1][0] if events else "N/A"
        location = events[0][3] if events else "Unknown"

        total_products = loaded_count + unloaded_count
        
        # Create a serializable dictionary of events, using index as key
        events_dict = {
            idx: {
                "timestamp": event[0],
                "status": event[1],
                "track_id": event[2],
                "location": event[3],
                "product_type": event[4]
            }
            for idx, event in enumerate(events)
        }
        
        return {
            "total_products": total_products,
            "operation_type": operation_type,
            "start_time": start_time,
            "end_time": end_time,
            "location": location,
            "detailed_product_counts": detailed_product_counts,
            "events": events_dict
        }