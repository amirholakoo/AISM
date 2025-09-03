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

import cv2
import streamlit as st
from ultralytics import YOLO

from config.warehouse_config import WarehouseConfig
from core.tracker import ProductTracker

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Handles video processing pipeline including model inference, tracking integration, and visualization.
    Manages the main processing loop and coordinates between YOLO model and ProductTracker.
    """
    
    def __init__(self):
        """Initialize video processor with default state."""
        self.model = None
        self.cap = None
        self.latest_frame = None
        self.is_running = False
        self.tracker = ProductTracker()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")

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

    def start_processing(self, source, weights_path, line_x, frame_skip, iou_thresh, conf_thresh, location, model_input_size):
        """
        Start video processing in a separate thread.
        
        Args:
            source: Video source (camera index or file path)
            weights_path: Path to YOLO weights
            line_x: X coordinate of counting line
            frame_skip: Number of frames to skip between processing
            iou_thresh: IoU threshold for tracking
            conf_thresh: Confidence threshold for detection
            location: Location name for this processing session
            model_input_size: The input size for the model (e.g., 640, 1280)
        """
        if not self.initialize_model(weights_path):
            return
            
        self.tracker.reset()
        self.is_running = True
        
        threading.Thread(
            target=self._processing_loop,
            args=(source, line_x, frame_skip, iou_thresh, conf_thresh, location, model_input_size),
            daemon=True
        ).start()

    def stop_processing(self):
        """
        Stop video processing and return session summary.
        
        Returns:
            Dict: Session summary with events and statistics
        """
        self.is_running = False
        time.sleep(0.5) 
        
        if self.cap:
            self.cap.release()
            self.cap = None
            
        events = list(self.tracker.events)
        counts = dict(self.tracker.counts)
        return self._create_session_summary(events, counts)

    def _processing_loop(self, source, line_x, frame_skip, iou_thresh, conf_thresh, location, model_input_size):
        """
        Main processing loop that runs in separate thread.
        
        Args:
            source: Video source
            line_x: X coordinate of counting line
            frame_skip: Frame skip interval
            iou_thresh: IoU threshold for tracking
            conf_thresh: Confidence threshold for detection
            location: Location name
            model_input_size: The input size for the model
        """
        while self.is_running:
            try:
                logger.info(f"Attempting to open video source with cv2.VideoCapture: {source}")
                self.cap = cv2.VideoCapture(source)

                if not self.cap.isOpened():
                    st.error(f"Cannot open source: {source}. Retrying in 5 seconds...")
                    logger.warning(f"Failed to open video source: {source}. isOpened() returned False.")
                    time.sleep(5)
                    continue
                
                logger.info(f"Successfully opened video source: {source}")
                frame_idx = 0
                while self.is_running and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if not ret:
                        st.warning("Stream ended or connection lost. Attempting to reconnect...")
                        logger.warning("Stream ended or connection lost. Breaking inner loop to reconnect.")
                        break
                        
                    frame_idx += 1
                    if frame_idx % frame_skip != 0:
                        continue
                        
                    timestamp_obj = datetime.now(WarehouseConfig.TIMEZONE)

                    results = self.model(frame, imgsz=model_input_size, verbose=False)[0]
                    boxes = results.boxes.xyxy.cpu().numpy()
                    class_ids = results.boxes.cls.cpu().numpy()
                    confidences = results.boxes.conf.cpu().numpy()

                    detections = [
                        (tuple(box.astype(int)), int(class_id))
                        for box, class_id, conf in zip(boxes, class_ids, confidences)
                        if conf >= conf_thresh and int(class_id) in WarehouseConfig.VALID_CLASSES
                    ]
                    
                    self.tracker.update_tracks(detections, frame, line_x, iou_thresh, timestamp_obj, location)
                    
                    self._draw_visualization(frame, line_x)
                    
                    self.latest_frame = frame.copy()

                    time.sleep(0.01)
                    
            except Exception as e:
                st.error(f"Processing loop error: {e}. Retrying in 5 seconds...")
                logger.error(f"An exception occurred in the processing loop: {e}", exc_info=True)
                time.sleep(5)
            finally:
                if self.cap:
                    self.cap.release()
        
        self.is_running = False
        logger.info("Processing stopped.")

    def _draw_visualization(self, frame, line_x):
        """
        Draw bounding boxes, track IDs, and statistics on frame.
        
        Args:
            frame: OpenCV frame to draw on
            line_x: X coordinate of counting line
        """
        height, _, _ = frame.shape
        cv2.line(frame, (line_x, 0), (line_x, height), (0, 0, 255), 2) 

        for track_id, track_data in self.tracker.tracks.items():
            x1, y1, x2, y2 = track_data['bbox']
            color = (0, 255, 0)  # Green color for bounding boxes

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        counts_text = f"Loaded: {self.tracker.counts['in']} | Unloaded: {self.tracker.counts['out']}"
        cv2.putText(frame, counts_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    def _create_session_summary(self, events, counts):
        """
        Create session summary from events and counts.
        
        Args:
            events: List of tracking events (timestamp, status, track_id, location, product_type)
            counts: Dictionary of in/out counts (overall)
            
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
            return {
                "total_products": 0,
                "operation_type": "none",
                "start_time": "N/A",
                "end_time": "N/A",
                "location": "Unknown",
                "detailed_product_counts": detailed_product_counts, 
                "events": {}
            }
        
        start_time = events[0][0] if events else "N/A" 
        end_time = events[-1][0] if events else "N/A"
        location = events[0][3] if events else "Unknown"
        
        total_products = max(loaded_count, unloaded_count)
        if loaded_count > unloaded_count:
            operation_type = "loaded"
        elif unloaded_count > loaded_count:
            operation_type = "unloaded"
        else:
            operation_type = "balanced" if loaded_count > 0 else "none" 

        formatted_events = {
            str(i): {
                "timestamp": event[0],
                "status": event[1],
                "track_id": event[2],
                "location": event[3],
                "product_type": event[4]
            }
            for i, event in enumerate(events)
        }
        
        return {
            "total_products": total_products, 
            "operation_type": operation_type, 
            "start_time": start_time,
            "end_time": end_time,
            "location": location,
            "detailed_product_counts": detailed_product_counts, 
            "events": formatted_events
        }