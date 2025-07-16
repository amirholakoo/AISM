"""
Advanced Product Tracking Module using Kalman Filter and State Machine.
"""
import collections
from datetime import datetime, timedelta
import logging
import os
from typing import Dict, List, Tuple
import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

from config.warehouse_config import WarehouseConfig

# Get a logger for this module
logger = logging.getLogger(__name__)

class Track:
    """
    Represents a single tracked object with state, Kalman Filter, and history.
    """
    def __init__(self, track_id: int, class_id: int, initial_bbox: Tuple[int, int, int, int], initial_confidence: float):
        self.id = track_id
        self.class_id = class_id
        self.state = 'TENTATIVE'  # TENTATIVE, CONFIRMED, COASTING, COUNTED
        
        self.kf = self.init_kalman_filter(initial_bbox)
        self.bbox = initial_bbox
        self.confidence = initial_confidence # From YOLO detection

        self.age = 0
        self.misses = 0
        self.history = collections.deque(maxlen=WarehouseConfig.TRACK_HISTORY_LEN)
        
        self.last_seen_timestamp = datetime.now(WarehouseConfig.TIMEZONE)
        self.counted_direction = None  # 'in' or 'out'
        self.counted_timestamp = None
        self.zone_entry_position = None  # Stores the (x, y) position where the track entered the counting zone

    @staticmethod
    def init_kalman_filter(bbox: Tuple[int, int, int, int]) -> KalmanFilter:
        """Initializes a Kalman Filter for a new track."""
        kf = KalmanFilter(dim_x=7, dim_z=4)
        # State: [x, y, s, r, dx, dy, ds] (center_x, center_y, scale, aspect_ratio, vel_x, vel_y, vel_s)
        kf.F = np.array([[1,0,0,0,1,0,0], [0,1,0,0,0,1,0], [0,0,1,0,0,0,1], [0,0,0,1,0,0,0],
                        [0,0,0,0,1,0,0], [0,0,0,0,0,1,0], [0,0,0,0,0,0,1]], dtype=float)
        kf.H = np.array([[1,0,0,0,0,0,0], [0,1,0,0,0,0,0], [0,0,1,0,0,0,0], [0,0,0,1,0,0,0]], dtype=float)
        
        # R: Measurement Noise. How much do we trust the YOLO detection?
        # Lower values mean we trust the detection more. Here, we trust x,y position
        # more than the scale/aspect_ratio of the box. This is a good starting point.
        kf.R[2:,2:] *= 10.
        
        # P: Initial State Covariance. How much uncertainty in our initial state?
        # High uncertainty in velocities because we have no idea how the object is moving initially.
        kf.P[4:,4:] *= 1000. 
        kf.P *= 10.

        # Q: Process Noise. How much do we trust the physics model (constant velocity)?
        # Higher values account for erratic movement (acceleration, turning).
        # This is the most important parameter to tune for your use case.
        # We're increasing noise on velocity estimates to make the filter more responsive.
        kf.Q[-1,-1] *= 0.1   # Noise on scale velocity
        kf.Q[4:6,4:6] *= 0.1 # Noise on x and y velocity

        # Correctly initialize the state vector from the measurement
        z = Track.bbox_to_z(bbox)
        kf.x[:4] = z
        return kf

    def predict(self):
        """Advances the state vector, updates the track's bbox to the prediction,
           and returns the predicted bounding box."""
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] = 0.
        self.kf.predict()
        self.age += 1
        self.misses += 1
        # Update the internal bbox to the new predicted position
        self.bbox = self.to_bbox()
        self.history.append(self.center)
        return self.bbox

    def update(self, bbox: Tuple[int, int, int, int], confidence: float):
        """Updates the Kalman Filter with a new detection."""
        self.bbox = bbox
        self.confidence = confidence
        self.misses = 0
        self.last_seen_timestamp = datetime.now(WarehouseConfig.TIMEZONE)
        self.history.append(self.center)
        self.kf.update(self.bbox_to_z(bbox))

    @property
    def center(self) -> Tuple[int, int]:
        """Calculates the center of the current bounding box."""
        return (self.bbox[0] + self.bbox[2]) // 2, (self.bbox[1] + self.bbox[3]) // 2

    def to_bbox(self) -> Tuple[int, int, int, int]:
        """Converts the Kalman Filter's state to a bounding box robustly."""
        x, y, s, r = self.kf.x[:4, 0]

        # Prevent invalid math from bad predictions by ensuring non-negativity
        s = max(0, s)
        r = max(0, r)

        w = np.sqrt(s * r)
        # Use a more stable formula for h and avoid division by zero
        h = np.sqrt(s / r) if r > 1e-6 else 0

        # Handle potential NaN values from sqrt of negative numbers if state becomes unstable
        if np.isnan(w): w = 0
        if np.isnan(h): h = 0

        return int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)

    @staticmethod
    def bbox_to_z(bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Converts a bounding box to the measurement vector [x, y, s, r]."""
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        x, y = x1 + w / 2, y1 + h / 2
        s = w * h
        r = w / h if h > 0 else 0
        return np.array([x, y, s, r]).reshape((4, 1))
        
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
        self.tracks: Dict[int, Track] = {}
        self.next_id: int = 0
        self.counts: Dict[str, int] = {'in': 0, 'out': 0}
        # Updated event structure: (timestamp, status, track_id, location, product_type)
        self.events: List[Tuple[str, str, int, str, str]] = [] 
        self.session_start_time: str = datetime.now(WarehouseConfig.TIMEZONE).strftime('%Y-%m-%d %H:%M:%S')
        # Cooldown is now global per event type ('loaded', 'unloaded')
        self.last_event_time: Dict[str, datetime] = {}

    def update_tracks(self, detections, frame, timestamp_obj, location):
        """
        Update tracking state with new detections using Kalman Filter and Hungarian Algorithm.
        """
        # 1. Predict new locations for existing tracks and update their history
        for tid in list(self.tracks.keys()):
            track = self.tracks[tid]
            track.predict()
            # The history is now updated inside track.predict(), so we don't do it here.
            if track.misses > WarehouseConfig.MAX_MISSES:
                logger.debug(f"Track {tid} removed due to high misses ({track.misses}).")
                if tid in self.tracks:
                    del self.tracks[tid]
        
        # 2. Associate detections with tracks and create new tracks
        unmatched_det_indices = set(range(len(detections)))
        
        if detections and len(self.tracks) > 0:
            det_bboxes = [d[0] for d in detections]
            track_bboxes = [t.to_bbox() for t in self.tracks.values()]
            
            iou_matrix = self._calculate_iou_matrix(track_bboxes, det_bboxes)
            
            if iou_matrix.size > 0:
                track_indices, det_indices = linear_sum_assignment(-iou_matrix) # Maximize IoU
                
                matched_indices = set()
                # Update matched tracks
                for track_idx, det_idx in zip(track_indices, det_indices):
                    # Check IoU threshold to prevent matching distant objects
                    if iou_matrix[track_idx, det_idx] >= WarehouseConfig.IOU_THRESHOLD:
                        tid = list(self.tracks.keys())[track_idx]
                        bbox, class_id, conf = detections[det_idx]
                        self.tracks[tid].update(bbox, conf)
                        self.tracks[tid].class_id = class_id # Update class ID on re-detection
                        matched_indices.add(det_idx)
                
                unmatched_det_indices = unmatched_det_indices - matched_indices

        # 3. Create new tracks for unmatched detections
        for det_idx in unmatched_det_indices:
            bbox, class_id, conf = detections[det_idx]
            if conf >= WarehouseConfig.MIN_DETECTION_CONFIDENCE:
                self._create_new_track(class_id, bbox, conf)

        # 4. Update track states and check for line crossings
        for tid in list(self.tracks.keys()):
            track = self.tracks[tid]
            self._update_track_state(track)
            if track.state in ['CONFIRMED', 'COASTING']:
                self._check_line_crossing(track, frame, timestamp_obj, location)

    def _create_new_track(self, class_id, bbox, confidence):
        new_track = Track(self.next_id, class_id, bbox, confidence)
        self.tracks[self.next_id] = new_track
        self.next_id += 1
        logger.debug(f"Created new TENTATIVE track {new_track.id}")

    def _update_track_state(self, track: Track):
        """Manages the lifecycle of a track through its state machine."""
        # Promote a TENTATIVE track to CONFIRMED if it has survived long enough.
        # Deletion is handled by the MAX_MISSES check in the main update loop.
        if track.state == 'TENTATIVE' and track.age >= WarehouseConfig.MIN_HITS_TO_CONFIRM:
            track.state = 'CONFIRMED'
            logger.debug(f"Track {track.id} promoted to CONFIRMED.")

        elif track.state == 'CONFIRMED' and track.misses > 0:
            track.state = 'COASTING'
            logger.debug(f"Track {track.id} moved to COASTING.")
        
        elif track.state == 'COASTING' and track.misses == 0:
            track.state = 'CONFIRMED'
            logger.debug(f"Track {track.id} returned to CONFIRMED.")

    def _calculate_iou_matrix(self, track_bboxes, det_bboxes):
        if not track_bboxes or not det_bboxes:
            return np.empty((0, 0))
        iou_matrix = np.zeros((len(track_bboxes), len(det_bboxes)), dtype=np.float32)
        for i, trk_bb in enumerate(track_bboxes):
            for j, det_bb in enumerate(det_bboxes):
                iou_matrix[i, j] = self._calc_iou(trk_bb, det_bb)
        return iou_matrix

    def _calc_iou(self, b1, b2):
        """Calculates IoU between two bounding boxes."""
        inter = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0])) * max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        uni = a1 + a2 - inter
        return inter / uni if uni > 0 else 0.0

    def _is_inside_zone(self, point: Tuple[int, int], zone: Tuple[int, int, int, int]) -> bool:
        """Checks if a point is inside the defined counting zone."""
        x, y = point
        zone_x1, zone_y1, zone_x2, zone_y2 = zone
        return zone_x1 <= x <= zone_x2 and zone_y1 <= y <= zone_y2

    def _check_line_crossing(self, track: Track, frame, ts_obj, loc):
        """
        Implements region-based counting logic. An object is counted when it exits
        the counting zone, based on its entry and exit points.
        """
        frame_h, frame_w, _ = frame.shape
        zone_x1 = int(frame_w * WarehouseConfig.COUNTING_ZONE_X_START_RATIO)
        zone_x2 = int(frame_w * WarehouseConfig.COUNTING_ZONE_X_END_RATIO)
        zone_y1 = int(frame_h * WarehouseConfig.COUNTING_ZONE_Y_START_RATIO)
        zone_y2 = int(frame_h * WarehouseConfig.COUNTING_ZONE_Y_END_RATIO)
        counting_zone = (zone_x1, zone_y1, zone_x2, zone_y2)

        current_center = track.center
        is_currently_inside = self._is_inside_zone(current_center, counting_zone)

        # Scenario 1: Track enters the counting zone for the first time
        if is_currently_inside and track.zone_entry_position is None:
            track.zone_entry_position = current_center
            logger.debug(f"Track {track.id} entered counting zone at {current_center}.")

        # Scenario 2: Track was inside and has now exited the zone
        elif not is_currently_inside and track.zone_entry_position is not None:
            entry_x, _ = track.zone_entry_position
            exit_x, _ = current_center
            
            # Basic check to prevent counting if object just wiggles out and back in
            if abs(exit_x - entry_x) < (frame_w * 0.05): # Less than 5% width travel
                logger.debug(f"Track {track.id} exited zone but moved too little to count. Resetting entry point.")
                track.zone_entry_position = None # Reset to allow re-entry
                return

            # Determine direction based on entry and exit points relative to zone center
            direction = None
            if exit_x > entry_x: # Moved left to right
                direction = 'out'
            else: # Moved right to left
                direction = 'in'

            # Get product name and status
            product_name = WarehouseConfig.PALLETE_CLASS_MAP.get(track.class_id, "Unknown")
            event_status = "unloaded" if direction == 'out' else "loaded"
            
            # Register the event and reset the tracking state for this zone
            self._register_event(track, direction, event_status, ts_obj, loc, product_name, frame)
            track.zone_entry_position = None # Reset after counting

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

    def _register_event(self, track: Track, direction: str, event_status: str, ts_obj, loc, product_name, frame):
        """Handles the logic for registering a counting event."""
        track.state = 'COUNTED'
        track.counted_direction = direction
        track.counted_timestamp = ts_obj
        self.counts[direction] += 1
        
        timestamp_str = ts_obj.strftime('%Y-%m-%d %H:%M:%S')
        self.events.append((timestamp_str, event_status, track.id, loc, product_name))
        logger.info(f"Event: {event_status}, Track ID: {track.id}, Product: {product_name}, Location: {loc}")
        self._save_snapshot(frame, track.bbox, ts_obj, event_status, track.id, product_name)
        
        # After a short period, allow the track to be counted again if it crosses back.
        # This is managed by the cooldown check in _check_line_crossing.
        # For this implementation, we reset its 'counted' status after cooldown.
        # A more advanced approach might use a separate state or timer.
        pass