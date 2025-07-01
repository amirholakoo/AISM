"""
Product Tracking Module
Contains the ProductTracker class responsible for object tracking and counting logic.
"""

from collections import deque
from datetime import datetime
from typing import Dict, List, Tuple
import logging
import os
import cv2

from config.warehouse_config import WarehouseConfig

# Get a logger for this module
logger = logging.getLogger(__name__)


class ProductTracker:
    """
    Handles multi-object tracking, trajectory management, and line crossing detection.
    Maintains object IDs across frames and counts entry/exit events.
    """
    
    def __init__(self):
        """Initialize tracker with empty state."""
        self.reset()
        self.snapshot_dir = "snapshots"
        if not os.path.exists(self.snapshot_dir):
            os.makedirs(self.snapshot_dir)
            logger.info(f"Created snapshots directory at: {self.snapshot_dir}")

    def reset(self):
        """Reset all tracking state for a new session."""
        self.tracks: Dict[int, Dict] = {}
        self.next_id: int = 0
        self.counts: Dict[str, int] = {'in': 0, 'out': 0}
        # Updated event structure: (timestamp, status, track_id, location, product_type)
        self.events: List[Tuple[str, str, int, str, str]] = [] 
        self.session_start_time: str = datetime.now(WarehouseConfig.TIMEZONE).strftime('%Y-%m-%d %H:%M:%S')
        # Cooldown is now global per event type ('loaded', 'unloaded')
        self.last_event_time: Dict[str, datetime] = {}

    def update_tracks(self, detections, frame, line_x, iou_thresh, timestamp_obj, location):
        """
        Update tracking state with new detections.
        
        Args:
            detections: List of (bbox, class_id) tuples
            frame: The current video frame for saving snapshots
            line_x: X coordinate of the counting line
            iou_thresh: IoU threshold for track association
            timestamp_obj: Current timestamp as a datetime object
            location: Current location name
        """
        matched = set()
        new_tracks = {}
        
        # Match existing tracks with new detections
        for tid, data in self.tracks.items():
            best_idx, _ = self._find_best_match(data['bbox'], detections, matched, iou_thresh)
            if best_idx is not None:
                bbox, cls_id = detections[best_idx]
                matched.add(best_idx)
                cx = (bbox[0] + bbox[2]) // 2
                data['bbox'] = bbox
                data['trajectory'].append(cx)
                data['class_history'].append(cls_id)
                self._check_line_crossing(data, tid, frame, line_x, timestamp_obj, location)
                new_tracks[tid] = data
        
        # Create new tracks for unmatched detections
        for idx, (bbox, cls_id) in enumerate(detections):
            if idx not in matched:
                new_tracks[self.next_id] = {
                    'bbox': bbox,
                    'trajectory': deque([(bbox[0] + bbox[2]) // 2], maxlen=5),
                    'class_history': deque([cls_id], maxlen=5),
                    'counted': {'in': False, 'out': False}
                }
                self.next_id += 1
        
        self.tracks = new_tracks

    def _find_best_match(self, b1, dets, matched, thresh):
        """
        Find the best matching detection for a given bounding box.
        
        Args:
            b1: Reference bounding box
            dets: List of detection bounding boxes
            matched: Set of already matched detection indices
            thresh: IoU threshold for matching
            
        Returns:
            Tuple of (best_index, best_score)
        """
        best_i, best_s = None, 0.0
        for i, (b2, _) in enumerate(dets):
            if i in matched:
                continue
            s = self._calc_iou(b1, b2)
            if s > best_s and s > thresh:
                best_s, best_i = s, i
        return best_i, best_s

    def _calc_iou(self, b1, b2):
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            b1, b2: Bounding boxes in format (x1, y1, x2, y2)
            
        Returns:
            IoU score as float
        """
        x1, y1, x2, y2 = b1
        x1g, y1g, x2g, y2g = b2
        
        # Calculate intersection
        inter = max(0, min(x2, x2g) - max(x1, x1g)) * max(0, min(y2, y2g) - max(y1, y1g))
        
        # Calculate union
        a1 = (x2 - x1) * (y2 - y1)
        a2 = (x2g - x1g) * (y2g - y1g)
        uni = a1 + a2 - inter
        
        return inter / uni if uni > 0 else 0.0

    def _save_snapshot(self, frame, bbox, timestamp_obj, status, track_id, product_name):
        """Saves a snapshot of the frame when an event occurs."""
        try:
            # Sanitize timestamp for filename
            timestamp_str = timestamp_obj.strftime('%Y-%m-%d_%H-%M-%S')
            filename = f"{timestamp_str}_{status}_ID-{track_id}_{product_name}.jpg"
            filepath = os.path.join(self.snapshot_dir, filename)
            
            # Draw the bounding box on a copy of the frame to not alter the live view
            frame_copy = frame.copy()
            x1, y1, x2, y2 = bbox
            color = (0, 0, 255)  # Red for snapshot highlight
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame_copy, f"Event: {status}", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            cv2.imwrite(filepath, frame_copy)
            logger.info(f"Saved snapshot: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}", exc_info=True)

    def _check_line_crossing(self, d, tid, frame, line_x, ts_obj, loc):
        """
        Check if a track has crossed the counting line and update counts.
        
        Args:
            d: Track data dictionary
            tid: Track ID
            frame: The current video frame for saving snapshots
            line_x: X coordinate of counting line
            ts_obj: Timestamp as a datetime object
            loc: Location name
        """
        if len(d['trajectory']) < 2:
            return

        current_class_id = d['class_history'][-1]
        prev_x, curr_x = d['trajectory'][-2], d['trajectory'][-1]
        
        # Ignore forklift_empty (class 0) for all counting events
        if current_class_id == WarehouseConfig.EMPTY_PALLETE_CLASS:
            return # No counting for empty forklifts

        # Proceed only if it's one of the loaded pallet classes
        # (e.g., sulfat, pack_material, neshaste)
        if current_class_id in WarehouseConfig.LOADED_PALLETE_CLASSES:
            product_name = WarehouseConfig.PALLETE_CLASS_MAP.get(current_class_id, "Unknown_Loaded_Type")
            timestamp_str = ts_obj.strftime('%Y-%m-%d %H:%M:%S')
            cooldown = WarehouseConfig.EVENT_COOLDOWN_SECONDS

            # Loaded: Right to Left (loaded forklift enters area with its load)
            if not d['counted']['in'] and prev_x > line_x and curr_x <= line_x:
                if 'loaded' in self.last_event_time and (ts_obj - self.last_event_time['loaded']).total_seconds() < cooldown:
                    return # Global cooldown for 'loaded' events is active

                d['counted']['in'] = True
                self.counts['in'] += 1
                self.events.append((timestamp_str, 'loaded', tid, loc, product_name))
                logger.info(f"Event: loaded, Track ID: {tid}, Product: {product_name}, Location: {loc}")
                self._save_snapshot(frame, d['bbox'], ts_obj, 'loaded', tid, product_name)
                self.last_event_time['loaded'] = ts_obj
            
            # Unloaded: Left to Right (loaded forklift exits area with its load)
            elif not d['counted']['out'] and prev_x < line_x and curr_x >= line_x:
                if 'unloaded' in self.last_event_time and (ts_obj - self.last_event_time['unloaded']).total_seconds() < cooldown:
                    return # Global cooldown for 'unloaded' events is active

                d['counted']['out'] = True
                self.counts['out'] += 1
                self.events.append((timestamp_str, 'unloaded', tid, loc, product_name))
                logger.info(f"Event: unloaded, Track ID: {tid}, Product: {product_name}, Location: {loc}")
                self._save_snapshot(frame, d['bbox'], ts_obj, 'unloaded', tid, product_name)
                self.last_event_time['unloaded'] = ts_obj