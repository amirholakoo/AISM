import os
import json
import cv2
import math
import threading
import queue
import time
from pathlib import Path
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from ultralytics import YOLO
OLD_LOG=False
LOCAL_TOTAL = 0
TOTAL_OFFSET=False
try:
    import torch
except ImportError:
    torch = None
# Faster shutter for high-speed objects
#os.environ["ANBAR_EXPOSURE_US"] = os.environ.get("ANBAR_EXPOSURE_US", "8000")
class Config:
    """Configuration class for the counting system"""
    # MODEL_PATH = "weights-416-v2/best-416-v2.onnx"
    MODEL_PATH = "weights/best-512_ncnn_model"      # 
    #MODEL_PATH = "best_ncnn_model_512_15_1"
    #MODEL_PATH = "best_ncnn_model_512_20_1"
    #MODEL_PATH = "best-512-70_ncnn_model"  #
    #MODEL_PATH ="weights/best-512-v2.onnx"
    #MODEL_PATH = "weights/best-free-v2.onnx"
    #MODEL_PATH ="weights/best.pt"
    #MODEL_PATH = "New/best-free-v2.onnx"
    #MODEL_PATH ="weights/best.pt"
    
    DETECTION_IMAGE_SIZE = 512
    CONFIDENCE_THRESHOLD = 0.4
    MAX_DETECTIONS = 30
    #VIDEO_INPUT_PATH = "picamera2"
    VIDEO_INPUT_PATH = "/home/admin/akhal/Mehdi Vahidi/22.mp4"
    VIDEO_OUTPUT_PATH = ""
    VIDEO_CODEC = "mp4v"

    PRIMARY_LINE_POSITION = 6 / 8 # 3/7
    SECONDARY_LINE_POSITION = 3 / 7 # 4/6
    # PRIMARY_LINE_POSITION = 5 /6 #2/7
    # SECONDARY_LINE_POSITION = 2 / 7 #5/6
    COUNTING_ZONE_MARGIN = 80
    FORKLIFT_CLASS_NAME = "forklift"
    BALE_CLASS_NAME = "akhal"
    EXCLUDE_CLASSES = []
    MATCH_DISTANCE = 70
    MAX_MISSED_FRAMES = 15
    SHOW_DISPLAY = True
    DISPLAY_WINDOW_NAME = "Akhal Bale Counter"
    QUEUE_SIZE = 32
    FPS_SMOOTHING_FRAMES = 30
    PROCESS_EVERY_N_FRAMES = 2
    DETECTION_PIPELINE_DEPTH = 2
    DETECTION_WORKERS = 2
    COLOR_BBOX = (0, 255, 0)
    COLOR_COUNTING_PRIMARY = (0, 0, 255)
    COLOR_COUNTING_SECONDARY = (0, 255, 255)
    COLOR_ZONE_BOUNDARY_PRIMARY = (0, 0, 255)
    COLOR_ZONE_BOUNDARY_SECONDARY = (0, 255, 255)
    COLOR_COUNTED_OBJECT = (0, 255, 255)
    COLOR_FPS_TEXT = (50, 255, 255)
    COLOR_PRIMARY_TEXT = (255, 255, 255)
    COLOR_SECONDARY_TEXT = (0, 255, 255)
    COLOR_TOTAL_TEXT = (0, 255, 0)
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE_LARGE = 0.9
    FONT_SCALE_MEDIUM = 0.7
    FONT_SCALE_SMALL = 0.5
    FONT_THICKNESS = 2
    VERBOSE = True
   
def apply_performance_settings():
    """Optimize threading for better performance"""
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
    cv2.setNumThreads(4)
    cv2.setUseOptimized(True)
    if torch is not None:
        torch.set_num_threads(4)
apply_performance_settings()
# ====================== Kalman Filter ======================
class KalmanFilter:
    """Simple 2D Constant Velocity Kalman Filter for centroid tracking"""
    def __init__(self, initial_x, initial_y):
        self.state = np.array([initial_x, initial_y, 0.0, 0.0], dtype=np.float32)
        self.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.Q = np.eye(4, dtype=np.float32)
        self.Q[0:2, 0:2] *= 0.05
        self.Q[2:4, 2:4] *= 0.5
        self.R = np.eye(2, dtype=np.float32) * 10.0
        self.P = np.eye(4, dtype=np.float32) * 100.0
    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state.copy()
    def update(self, measurement):
        z = np.array(measurement, dtype=np.float32).reshape(2, 1)
        state_reshaped = self.state.reshape(4, 1)
        y = z - (self.H @ state_reshaped)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = (state_reshaped + K @ y).flatten()
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.state[:2].astype(int)
    def get_predicted_position(self):
        return self.state[:2].astype(int)
# ====================== FPS Calculator ================================
class FPSCalculator:
    def __init__(self, smoothing_frames=30):
        self.smoothing_frames = smoothing_frames
        self.timestamps = deque(maxlen=smoothing_frames)
    def end_frame(self):
        now = time.time()
        self.timestamps.append(now)
        if len(self.timestamps) >= 2:
            duration = self.timestamps[-1] - self.timestamps[0]
            if duration > 0:
                return (len(self.timestamps) - 1) / duration
        return 0.0
    def get_current_fps(self):
        return self.end_frame()
# ====================== Frame Annotator ===============================
class FrameAnnotator:
    def __init__(self, config, primary_line_x, secondary_line_x):
        self.config = config
        self.primary_line_x = primary_line_x
        self.secondary_line_x = secondary_line_x
        self.margin = config.COUNTING_ZONE_MARGIN
    def annotate_frame(self, frame, tracked_objects, counts, fps):
        annotated_frame = frame.copy()
        self._draw_tracks(annotated_frame, tracked_objects)
        self._draw_counting_lines(annotated_frame)
        self._draw_info_overlay(annotated_frame, counts, fps)
        return annotated_frame
    def _draw_tracks(self, frame, tracked_objects):
        for track in tracked_objects:
            bbox = track.get("bbox")
            if bbox is None:
                continue
            color = self.config.COLOR_BBOX
            if track["counted_primary"] or track["counted_secondary"]:
                color = self.config.COLOR_COUNTED_OBJECT
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (track["cx"], track["cy"]), 5, color, -1)
            label = f'ID:{track["track_id"]} Conf:{track["conf"]:.2f}'
            cv2.putText(frame, label, (x1, max(y1 - 5, 0)), self.config.FONT,
                        self.config.FONT_SCALE_SMALL, color, self.config.FONT_THICKNESS)
    def _draw_counting_lines(self, frame):
        height = frame.shape[0]
        cv2.line(frame, (self.primary_line_x, 0), (self.primary_line_x, height),
                 self.config.COLOR_COUNTING_PRIMARY, 3)
        cv2.line(frame, (self.primary_line_x - self.margin, 0),
                 (self.primary_line_x - self.margin, height),
                 self.config.COLOR_ZONE_BOUNDARY_PRIMARY, 1)
        cv2.line(frame, (self.primary_line_x + self.margin, 0),
                 (self.primary_line_x + self.margin, height),
                 self.config.COLOR_ZONE_BOUNDARY_PRIMARY, 1)
        cv2.line(frame, (self.secondary_line_x, 0), (self.secondary_line_x, height),
                 self.config.COLOR_COUNTING_SECONDARY, 3)
        cv2.line(frame, (self.secondary_line_x - self.margin, 0),
                 (self.secondary_line_x - self.margin, height),
                 self.config.COLOR_ZONE_BOUNDARY_SECONDARY, 1)
        cv2.line(frame, (self.secondary_line_x + self.margin, 0),
                 (self.secondary_line_x + self.margin, height),
                 self.config.COLOR_ZONE_BOUNDARY_SECONDARY, 1)
    def _draw_info_overlay(self, frame, counts, fps):
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), self.config.FONT,
                    self.config.FONT_SCALE_MEDIUM, self.config.COLOR_FPS_TEXT, self.config.FONT_THICKNESS)
        cv2.putText(frame, f'Primary: {counts["primary"]}', (10, 65), self.config.FONT,
                    self.config.FONT_SCALE_MEDIUM, self.config.COLOR_PRIMARY_TEXT, self.config.FONT_THICKNESS)
        cv2.putText(frame, f'Secondary: {counts["secondary"]}', (10, 100), self.config.FONT,
                    self.config.FONT_SCALE_MEDIUM, self.config.COLOR_SECONDARY_TEXT, self.config.FONT_THICKNESS)
        cv2.putText(frame, f'Total: {counts["total"]}', (10, 135), self.config.FONT,
                    self.config.FONT_SCALE_MEDIUM, self.config.COLOR_TOTAL_TEXT, self.config.FONT_THICKNESS)
        live_count = counts.get("live", 1)
        cv2.putText(frame, f"Live: {live_count}", (10, 170), self.config.FONT,
                    self.config.FONT_SCALE_MEDIUM, self.config.COLOR_TOTAL_TEXT, self.config.FONT_THICKNESS)
        last_obj = counts.get("last_non_forklift_primary_class")
        if last_obj:
            cv2.putText(frame, f"Last: {last_obj}", (10, 205), self.config.FONT,
                        self.config.FONT_SCALE_MEDIUM, self.config.COLOR_PRIMARY_TEXT, self.config.FONT_THICKNESS)
        cv2.putText(frame, f"Forklift In: {counts.get('forklift_entry', False)}", (10, 240),
                    self.config.FONT, self.config.FONT_SCALE_MEDIUM,
                    self.config.COLOR_PRIMARY_TEXT, self.config.FONT_THICKNESS)
        cv2.putText(frame, f"Forklift out: {counts.get('forklift_exit', False)}", (10, 280),
                    self.config.FONT, self.config.FONT_SCALE_MEDIUM,
                    self.config.COLOR_PRIMARY_TEXT, self.config.FONT_THICKNESS)            
# ====================== Object Detector ======================
class ObjectDetector:
    def __init__(self, config):
        self.imgsz = config.DETECTION_IMAGE_SIZE
        self.conf_threshold = config.CONFIDENCE_THRESHOLD
        self.max_detections = getattr(config, "MAX_DETECTIONS", 30)
        self.exclude_classes = [cls.lower() for cls in config.EXCLUDE_CLASSES]
        self.model_path = Path(config.MODEL_PATH)
        self.model = YOLO(str(self.model_path), task="detect")
        self.use_torch_context = torch is not None
        self.forklift_class_name = getattr(config, "FORKLIFT_CLASS_NAME", "forklift").lower()
    def detect(self, frame):
        predict_kwargs = {
            "verbose": False,
            "imgsz": self.imgsz,
            "conf": self.conf_threshold,
            "max_det": self.max_detections,
        }
        if self.use_torch_context:
            with torch.inference_mode():
                results = self.model(frame, **predict_kwargs)
        else:
            results = self.model(frame, **predict_kwargs)
        detections = []
        if results:
            result = results[0]
            if hasattr(result, "boxes") and result.boxes is not None and len(result.boxes) > 0:
                detections.extend(self._parse_box_detections(result))
        if len(detections) > 1:
            detections = self._filter_duplicate_forklift_detections(detections)
        return detections
    def _to_numpy(self, value):
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            return value.numpy()
        return np.asarray(value)
    def _parse_box_detections(self, result):
        detections = []
        boxes = result.boxes
        xyxy = self._to_numpy(boxes.xyxy)
        classes = self._to_numpy(boxes.cls).astype(int)
        confidences = self._to_numpy(boxes.conf)
        for box, cls_id, conf in zip(xyxy, classes, confidences):
            class_name = self.model.names[int(cls_id)].lower()
            if class_name in self.exclude_classes:
                continue
            x1, y1, x2, y2 = [int(coord) for coord in box]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            detections.append({
                "cx": cx, "cy": cy, "bbox": (x1, y1, x2, y2),
                "class_name": class_name, "conf": float(conf),
            })
        return detections
    def _filter_duplicate_forklift_detections(self, detections):
        forklifts = [d for d in detections if d["class_name"] == self.forklift_class_name]
        if len(forklifts) <= 1:
            return detections
        best = max(forklifts, key=lambda det: det["conf"])
        filtered = [d for d in detections if d["class_name"] != self.forklift_class_name]
        filtered.append(best)
        return filtered
# ====================== Object Tracker with Kalman ======================
class ObjectTracker:
    def __init__(self, primary_line_x, secondary_line_x, config):
        self.primary_line_x = primary_line_x
        self.secondary_line_x = secondary_line_x
        self.counting_zone_margin = config.COUNTING_ZONE_MARGIN
        self.match_distance = config.MATCH_DISTANCE
        self.max_missed_frames = config.MAX_MISSED_FRAMES
        self.forklift_class_name = getattr(config, "FORKLIFT_CLASS_NAME", "forklift").lower()
        self.bale_class_name = getattr(config, "BALE_CLASS_NAME", "akhal").lower()
        self.tracks = {}
        self.next_track_id = 0
        # Batch counts - reset only on forklift exit
        self.primary_count = 0
        self.secondary_count = 0
        self.total_count = 0  # Cumulative confirmed total (after cycle completion)
        # Forklift state
        self.forklift_inside = False
        self.forklift_entry_count = False
        self.forklift_exit_count = False
        self.bale_entry_count = 0
        self.bale_exit_count = 0
        self.last_non_forklift_primary_class = ""
        self.current_live_bales = 1
        self.events = []
    def _finalize_cycle_on_forklift_exit(self):
        """On forklift exit: add max(primary, secondary) to total and reset both"""
        if self.primary_count > 0 or self.secondary_count > 0:
            batch_size = max(self.primary_count, self.secondary_count)
            self.total_count += batch_size
            self.events.append({
                "event_type": "cycle_completed",
                "primary": self.primary_count,
                "secondary": self.secondary_count,
                "added_to_total": batch_size
            })
            self.primary_count = 0
            self.secondary_count = 0
    def update(self, detections, frame_index):
        self.events = []
        for track in self.tracks.values():
            track["matched"] = False
        for track in self.tracks.values():
            track["kalman"].predict()
        tracked_visuals = []
        self.current_live_bales = 1
        for det in detections:
            cx, cy = det["cx"], det["cy"]
            best_id = self._find_best_track(cx, cy)
            if best_id is None:
                best_id = self._create_track(cx, cy, det["class_name"], frame_index)
            track = self.tracks[best_id]
            track["matched"] = True
            track["missed"] = 0
            track["last_seen"] = frame_index
            smooth_pos = track["kalman"].update([cx, cy])
            smooth_cx, smooth_cy = smooth_pos[0], smooth_pos[1]
            track["prev_x_prev"] = track["prev_x"]
            track["prev_y_prev"] = track["prev_y"]
            track["prev_x"] = smooth_cx
            track["prev_y"] = smooth_cy
            track["class_name"] = det["class_name"]
            track["bbox"] = det["bbox"]
            track["conf"] = det["conf"]
            self._check_crossings(best_id, track)
            tracked_visuals.append({
                "track_id": best_id,
                "cx": smooth_cx,
                "cy": smooth_cy,
                "bbox": det["bbox"],
                "class_name": det["class_name"],
                "conf": det["conf"],
                "counted_primary": track["counted_primary"],
                "counted_secondary": track["counted_secondary"],
            })
        self._purge_stale_tracks()
        # <<< Live Total: previous confirmed + current batch max >>>
        current_batch_max = max(self.primary_count, self.secondary_count)
        displayed_total = self.total_count + current_batch_max
        return {
            "tracks": tracked_visuals,
            "counts": {
                "primary": self.primary_count,
                "secondary": self.secondary_count,
                "total": displayed_total,  # Live and real-time total
                "forklift_entry": self.forklift_entry_count,
                "forklift_exit": self.forklift_exit_count,
                "bale_entry": self.bale_entry_count,
                "bale_exit": self.bale_exit_count,
                "live": self.current_live_bales,
                "last_non_forklift_primary_class": self.last_non_forklift_primary_class,
            },
            "events": list(self.events),
        }
    def _find_best_track(self, cx, cy):
        best_id = None
        best_distance = self.match_distance
        for track_id, track in self.tracks.items():
            pred_x, pred_y = track["kalman"].get_predicted_position()
            distance = math.hypot(cx - pred_x, cy - pred_y)
            if distance < best_distance:
                best_distance = distance
                best_id = track_id
        return best_id
    def _create_track(self, cx, cy, class_name, frame_index):
        self.next_track_id += 1
        class_name_lc = str(class_name).lower()
        is_forklift = class_name_lc == self.forklift_class_name
        counted_primary = cx <= self.primary_line_x if not is_forklift else False
        counted_secondary = cx <= self.secondary_line_x if not is_forklift else False
        kf = KalmanFilter(cx, cy)
        self.tracks[self.next_track_id] = {
            "prev_x": cx, "prev_y": cy,
            "prev_x_prev": cx, "prev_y_prev": cy,
            "class_name": class_name,
            "counted_primary": counted_primary,
            "counted_secondary": counted_secondary,
            "forklift_entry_done": False,
            "forklift_exit_done": False,
            "matched": True,
            "missed": 0,
            "last_seen": frame_index,
            "bbox": None,
            "conf": 0.0,
            "kalman": kf,
        }
        return self.next_track_id

    def _check_crossings(self, track_id, track):
        prev_prev_x = track["prev_x_prev"]
        prev_x = track["prev_x"]
        if prev_prev_x is None:
            print("############# >>>>>>>>>>>>>>>>> fucking kalman <<<<<<<<<<<<<< #############")
            return

        class_name = str(track["class_name"]).lower()
        is_forklift = class_name == self.forklift_class_name
        is_bale = class_name == self.bale_class_name

        primary_rtl = prev_prev_x > self.primary_line_x and prev_x <= self.primary_line_x
        primary_ltr = prev_prev_x < self.primary_line_x and prev_x >= self.primary_line_x
        secondary_rtl = prev_prev_x > self.secondary_line_x and prev_x <= self.secondary_line_x

        if is_forklift:
            if primary_rtl and abs(prev_x - self.primary_line_x) <= self.counting_zone_margin \
                    and not track.get("forklift_entry_done", False):
                track["forklift_entry_done"] = True
                track["counted_primary"] = True
                self.forklift_inside = True
                self.forklift_entry_count = True
                self.forklift_exit_count = False
                self.current_live_bales = max(self.current_live_bales, 1)
                # Event_log_temp.append({
                #      "track_id": track_id,
                #     "class_name": class_name,
                #     "line": "primary",
                #     "direction": "right_to_left",
                #     "event_type": "forklift_entry"
                # })
                # print ("===========is_forklift:rtl ", Event_log_temp)
                # # print ("===========events-temp ", self.events)
                # print(str(len (self.events)))
                self.events.append({
                    "track_id": track_id,
                    "class_name": class_name,
                    "line": "primary",
                    "direction": "right_to_left",
                    "event_type": "forklift_entry"
                })

            if primary_ltr and abs(prev_x - self.primary_line_x) <= self.counting_zone_margin \
                    and not track.get("forklift_exit_done", False):
                track["forklift_exit_done"] = True
                track["counted_primary"] = True
                self.forklift_inside = False
                self.forklift_exit_count = True
                self.forklift_entry_count = False
                self.current_live_bales = max(self.current_live_bales, 1)

                self.events.append({
                    "track_id": track_id,
                    "class_name": class_name,
                    "line": "primary",
                    "direction": "left_to_right",
                    "event_type": "forklift_exit"
                })
                # print ("===========is_forklift:ltr ", Event_log_temp)
                # # print ("===========events-temp ", self.events)
                # print(str(len (self.events)))
                self._finalize_cycle_on_forklift_exit()

        else:
            # Crossing secondary line
            if not track["counted_secondary"] and secondary_rtl and \
                    abs(prev_x - self.secondary_line_x) <= self.counting_zone_margin:
                self.secondary_count += 1
                track["counted_secondary"] = True
                if is_bale:
                    self.bale_entry_count += 1
                    # self.bale_exit_count += 1
                    self.forklift_entry_count = True
                    self.forklift_exit_count = False

                else:
                    self.forklift_exit_count = False
                    self.forklift_entry_count = True
                
                self.current_live_bales = 1

                # Event_log_temp.append({
                #      "track_id": track_id,
                #     "class_name": class_name,
                #     "line": "secondary",
                #     "direction": "right_to_left",
                #     "event_type": "bale_entry" if is_bale else "other_entry"
                # })
                # print ("===========is_bale:secondary ", Event_log_temp)
                # # print ("===========events-temp ", self.events)
                # print(str(len (self.events)))
                self.events.append({
                    "track_id": track_id,
                    "class_name": class_name,
                    "line": "secondary",
                    "direction": "right_to_left",
                    "event_type": "bale_entry" if is_bale else "other_entry"
                })
            
             # Crossing primary line (for bale or other objects)
            # if not self.current_live_bales and not track["counted_primary"] and primary_rtl and \
            #         abs(prev_x - self.primary_line_x) <= self.counting_zone_margin:
            if not track["counted_primary"] and primary_rtl and \
                    abs(prev_x - self.primary_line_x) <= self.counting_zone_margin:

                self.primary_count += 1
                track["counted_primary"] = True
                if is_bale:
                    self.bale_entry_count += 1
                    self.forklift_entry_count = True
                    self.forklift_exit_count = False
                else:
                    self.forklift_exit_count = True
                    self.forklift_entry_count = False

                self.last_non_forklift_primary_class = class_name
                self.current_live_bales = 1

                # Event_log_temp.append({
                #      "track_id": track_id,
                #     "class_name": class_name,
                #     "line": "primary",
                #     "direction": "right_to_left",
                #     "event_type": "bale_entry" if is_bale else "other_entry"
                # })


                self.events.append({
                    "track_id": track_id,
                    "class_name": class_name,
                    "line": "primary",
                    "direction": "right_to_left",
                    "event_type": "bale_entry" if is_bale else "other_entry"
                })
   
                # print ("===========is_bale: primary ", Event_log_temp)
                # # print ("===========events-temp ", self.events)
                # print(str(len (self.events)))



    # def _check_crossings(self, track_id, track):
    #     prev_prev_x = track["prev_x_prev"]
    #     prev_x = track["prev_x"]
    #     if prev_prev_x is None:
    #         return
    #     class_name = str(track["class_name"]).lower()
    #     is_forklift = class_name == self.forklift_class_name
    #     is_bale = class_name == self.bale_class_name
    #     primary_rtl = prev_prev_x > self.primary_line_x and prev_x <= self.primary_line_x
    #     primary_ltr = prev_prev_x < self.primary_line_x and prev_x >= self.primary_line_x
    #     secondary_rtl = prev_prev_x > self.secondary_line_x and prev_x <= self.secondary_line_x
    #     if is_forklift:
    #         if primary_rtl and abs(prev_x - self.primary_line_x) <= self.counting_zone_margin and not track.get("forklift_entry_done", False):
    #             track["forklift_entry_done"] = True
    #             track["counted_primary"] = True
    #             self.forklift_inside = True
    #             self.forklift_entry_count = True
    #             self.forklift_exit_count = False
    #             self.current_live_bales = max(self.current_live_bales, 1)
    #             self.events.append({"track_id": track_id, "event_type": "forklift_entry"})
    #         if primary_ltr and abs(prev_x - self.primary_line_x) <= self.counting_zone_margin and not track.get("forklift_exit_done", False):
    #             track["forklift_exit_done"] = True
    #             track["counted_primary"] = True
    #             self.forklift_inside = False
    #             self.forklift_exit_count = True
    #             self.forklift_entry_count = False
    #             self.current_live_bales = max(self.current_live_bales, 1)
    #             self.events.append({"track_id": track_id, "event_type": "forklift_exit"})
    #             self._finalize_cycle_on_forklift_exit()
    #     else:
    #         if not track["counted_primary"] and primary_rtl and abs(prev_x - self.primary_line_x) <= self.counting_zone_margin:
    #             self.primary_count += 1
    #             track["counted_primary"] = True
    #             if is_bale:
    #                 self.bale_entry_count += 1
    #                 self.forklift_entry_count = True
    #                 self.forklift_exit_count = False
    #             else:
    #                 self.forklift_exit_count = True
    #                 self.forklift_entry_count = False
    #             self.last_non_forklift_primary_class = class_name
    #             self.current_live_bales = 1
    #             self.events.append({"track_id": track_id, "event_type": "bale_entry" if is_bale else "other_entry"})
    #         if not track["counted_secondary"] and secondary_rtl and abs(prev_x - self.secondary_line_x) <= self.counting_zone_margin:
    #             self.secondary_count += 1
    #             track["counted_secondary"] = True
    #             if is_bale:
    #                 self.bale_exit_count += 1
    #                 self.forklift_entry_count = False
    #                 self.forklift_exit_count = True
    #             else:
    #                 self.forklift_exit_count = False
    #                 self.forklift_entry_count = True
    #             self.events.append({"track_id": track_id, "event_type": "bale_exit" if is_bale else "other_exit"})
    def _purge_stale_tracks(self):
        stale_ids = [tid for tid, t in self.tracks.items() if not t["matched"]]
        for tid in stale_ids:
            track = self.tracks[tid]
            track["missed"] += 1
            if track["missed"] > self.max_missed_frames:
                del self.tracks[tid]
# ====================== Video Reader ======================
class VideoReader:
    """Thread-safe video reader class"""
    def __init__(self, video_path, queue_size=128):
        self.video_path = video_path
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.thread = None
        self.cap = None
        self.picam2 = None
        try:
            from picamera2 import Picamera2
            PICAMERA2_AVAILABLE = True
        except ImportError:
            PICAMERA2_AVAILABLE = False
        if video_path == "picamera2" and PICAMERA2_AVAILABLE:
            self._init_picamera2()
        else:
            self._init_opencv_capture(video_path)
    def _init_opencv_capture(self, source):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file or camera: {source}")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

   # def _init_picamera2(self):
    #    try:
         #   from picamera2 import Picamera2
      #      self.picam2 = Picamera2()
      #      # Optimized settings for Camera Module 3
       #     config = self.picam2.create_video_configuration(
      #          main={"size": (1920, 1080), "format": "RGB888"},
                # Low resolution for LoF (optional)
                # lores={"size": (640, 480), "format": "YUV420"}
     #       )
    #        self.picam2.configure(config)
     #       # Manual settings for better lighting (optional)
   #         self.picam2.set_controls({
   #             "AwbMode": 1,           # Auto White Balance
      #          "ExposureTime": 8000,   # Fast shutter (us) - same as in your code
    #            "AnalogueGain": 4.0,
       #         "Brightness": 0.1,
        #        "Contrast": 1.2
        #    })
       #     self.picam2.start()
     #       time.sleep(2)  # Extra time for stabilization
    #        self.width = 1920
    #        self.height = 1080
    #        self.fps = 30.0
    #        self.total_frames = 0
    #    except Exception as e:
   #         print(f"PiCamera2 failed: {e}")
  #          self._init_opencv_capture(0)
        
    def _init_picamera2(self):
         try:
             from picamera2 import Picamera2
             self.picam2 = Picamera2()
             config = self.picam2.create_preview_configuration(main={"size": (1920, 1080), "format": "RGB888"})
             self.picam2.configure(config)
             self.picam2.start()
             time.sleep(1)
             self.width = 1920
             self.height = 1080
             self.fps = 30.0
             self.total_frames = 0
         except Exception as e:
             print(f"PiCamera2 failed: {e}")
             self._init_opencv_capture(0)
             
    def start(self):
        self.thread = threading.Thread(target=self._read_frames, daemon=True)
        self.thread.start()
        return self
    def _read_frames(self):
        frame_number = 0
        while not self.stopped:
            if not self.frame_queue.full():
                if self.picam2 is not None:
                    frame = self.picam2.capture_array()
                    ret = frame is not None
                else:
                    ret, frame = self.cap.read()
                if not ret or frame is None:
                    self.stopped = True
                    break
                frame = frame.copy()
                frame_number += 1
                self.frame_queue.put({"frame": frame, "frame_number": frame_number, "timestamp": time.time()})
            else:
                time.sleep(0.001)
        if self.cap is not None:
            self.cap.release()
        if self.picam2 is not None:
            self.picam2.stop()
    def read(self):
        try:
            return self.frame_queue.get(timeout=1.0)
        except queue.Empty:
            return None
    def more(self):
        return not self.stopped or not self.frame_queue.empty()
    def stop(self):
        self.stopped = True
        if self.thread is not None:
            self.thread.join()
    def get_properties(self):
        return {"width": self.width, "height": self.height, "fps": self.fps, "total_frames": self.total_frames}
# ====================== Video Writer ======================
class VideoWriter:
    """Thread-safe video writer class"""
    def __init__(self, output_path, fps, frame_size, codec="mp4v", queue_size=128):
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        if not self.writer.isOpened():
            raise ValueError(f"Cannot open video writer: {output_path}")
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.thread = None
        self.frames_written = 0
    def start(self):
        self.thread = threading.Thread(target=self._write_frames, daemon=True)
        self.thread.start()
        return self
    def _write_frames(self):
        while not self.stopped or not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get(timeout=1.0)
                if frame is not None:
                    self.writer.write(frame)
                    self.frames_written += 1
            except queue.Empty:
                continue
        self.writer.release()
    def write(self, frame):
        if not self.stopped:
            try:
                self.frame_queue.put(frame, timeout=1.0)
            except queue.Full:
                print("Warning: Video writer queue full, dropping frame")
    def stop(self):
        self.stopped = True
        if self.thread is not None:
            self.thread.join()
    def get_frames_written(self):
        return self.frames_written
# ====================== Main Application ======================
class BaleCountingApp:
    def __init__(self, config: Config):
        self.config = config
        self.video_reader = VideoReader(config.VIDEO_INPUT_PATH, config.QUEUE_SIZE)
        props = self.video_reader.get_properties()
        self.primary_line_x = int(props["width"] * config.PRIMARY_LINE_POSITION)
        self.secondary_line_x = int(props["width"] * config.SECONDARY_LINE_POSITION)
        self.detector = ObjectDetector(config)
        self.tracker = ObjectTracker(self.primary_line_x, self.secondary_line_x, config)
        self.annotator = FrameAnnotator(config, self.primary_line_x, self.secondary_line_x)
        self.fps_calculator = FPSCalculator(config.FPS_SMOOTHING_FRAMES)
        self.pipeline_depth = max(1, getattr(config, "DETECTION_PIPELINE_DEPTH", 1))
        worker_count = max(self.pipeline_depth, getattr(config, "DETECTION_WORKERS", 1))
        self.executor = ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="detector")
        self.pending_jobs = deque()
        self.video_writer = None
        if config.VIDEO_OUTPUT_PATH:
            self.video_writer = VideoWriter(
                config.VIDEO_OUTPUT_PATH, props["fps"],
                (props["width"], props["height"]), codec=config.VIDEO_CODEC,
                queue_size=config.QUEUE_SIZE
            ).start()
        self.processed_frames = 0
        self.total_frames = props["total_frames"]
        self.frame_number = 0
        self._stop_requested = False
        self._is_running = False
        self.report_events = []
        self.last_counts = {k: 0 for k in ["primary", "secondary", "total", "live"]}
        self.start_timestamp = None
    def request_stop(self):
        """Signal the app to stop processing as soon as possible."""
        self._stop_requested = True
    def run(self):
        print("Starting bale counting...")
        print(f"   Input: {self.config.VIDEO_INPUT_PATH}")
        print(f"   Model: {self.config.MODEL_PATH}")
        print(f"   Frame skip: {self.config.PROCESS_EVERY_N_FRAMES}")
        start_time = time.time()
        self.start_timestamp = time.time()
        self._stop_requested = False
        self._is_running = True
        self.video_reader.start()
        self._fill_detection_pipeline()
        try:
            while True:
                if self._stop_requested:
                    print("\n Stop requested, finishing current work...")
                    break
                if not self.pending_jobs and not self._fill_detection_pipeline():
                    break
                if not self.pending_jobs:
                    continue
                frame_data, future = self.pending_jobs.popleft()
                if frame_data is None:
                    continue
                frame = frame_data["frame"]
                self.frame_number = frame_data["frame_number"]
                self._fill_detection_pipeline()
                self.fps_calculator.end_frame()
                try:
                    detections = future.result()
                except Exception as exc:
                    print(f"\ Detection error on frame {self.frame_number}: {exc}")
                    continue
                tracking_info = self.tracker.update(detections, self.frame_number)
                fps_value = self.fps_calculator.get_current_fps()
                annotated = self.annotator.annotate_frame(frame, tracking_info["tracks"], tracking_info["counts"], fps_value)
                self._handle_reporting(annotated, tracking_info["counts"], tracking_info.get("events", []))
                if self.video_writer:
                    self.video_writer.write(annotated)
                if self.config.SHOW_DISPLAY:
                    display = cv2.resize(annotated, (640, 360))
                    cv2.imshow(self.config.DISPLAY_WINDOW_NAME, display)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        raise KeyboardInterrupt
                self.processed_frames += 1
                if self.processed_frames % 30 == 0:
                    counts = tracking_info["counts"]
                    progress = (
                        (self.frame_number / self.total_frames) * 100
                        if self.total_frames > 0
                        else 0
                    )
                    print(
                        f" {self.frame_number}/{self.total_frames} ({progress:.1f}%) | "
                        f"Processed: {self.processed_frames} | "
                        f"Primary: {counts['primary']} | Secondary: {counts['secondary']} | Total : {counts['total']}"
                    )
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            elapsed = time.time() - start_time
            self._cleanup(elapsed)
            self._is_running = False
    def _fill_detection_pipeline(self):
        scheduled = False
        while len(self.pending_jobs) < self.pipeline_depth:
            job_added = self._schedule_detection_job()
            if not job_added:
                break
            scheduled = True
        return scheduled
    def _schedule_detection_job(self):
        while self.video_reader.more():
            frame_data = self.video_reader.read()
            if frame_data is None:
                continue
            frame_number = frame_data["frame_number"]
            if frame_number % self.config.PROCESS_EVERY_N_FRAMES != 0:
                continue
            future = self.executor.submit(self.detector.detect, frame_data["frame"])
            self.pending_jobs.append((frame_data, future))
            return True
        return False
    def _handle_reporting(self, frame, counts, events):
        global OLD_LOG
        global TOTAL_OFFSET
        global LOCAL_TOTAL
        report_dir = "result"
        temp_dir = "temp"
        os.makedirs(report_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)
        increased = (
            counts["primary"] > self.last_counts.get("primary", 0)
            or counts["secondary"] > self.last_counts.get("secondary", 0)
            or counts["total"] > self.last_counts.get("total", 0)
        )
        snapshot_path = None
        if increased:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"snapshot_{self.frame_number}_{counts['total']}_{timestamp}.jpg"
            snapshot_path = os.path.join(report_dir, filename)
            cv2.imwrite(snapshot_path, frame)
            self.report_events.append({
                "frame_number": self.frame_number,
                "timestamp": time.time(),
                "counts": dict(counts),
                "snapshot_path": snapshot_path,
                "live": counts.get("live", 0),
            })
        events_log_path = os.path.join(temp_dir, "events_log.json")
        # print("======================================================= before events in",events)
        if events:
            # print("======================================================= events in")
            try:
                with open(events_log_path, "a", encoding="utf-8") as log_file:
                    for event in events:
                        event_type = event.get("event_type") or ""
                        class_name = str(event.get("class_name", "")).lower()
                        if not (event_type.startswith("forklift_") or event_type.startswith("bale_")):
                            # print("======================================================= is not bale or forklift")
                            continue
                        event_timestamp = time.time()
                        json_payload = {
                            "timestamp": event_timestamp,
                            "frame_number": event.get("frame_number", self.frame_number),
                            "track_id": event.get("track_id"),
                            "class_name": class_name,
                            "line": event.get("line"),
                            "direction": event.get("direction"),
                            "event_type": event_type,
                            "snapshot_path": snapshot_path or "",
                            "counts": dict(counts),
                            "live": counts.get("live", 1),
                            "device_id": "anbar_akhal",
                        }
                        print(">>>>>>>>>>>>>>>>>>>>>>>",json_payload)
                        if json_payload["counts"].get("live", 0) != 0:

                            print("======================================================= Before_log_line")
                            
                            do_not_log = False
                            #print("=========Json_log_temp  before and total offset",do_not_log)
                            print("======================================================= total offset 0 _",TOTAL_OFFSET,LOCAL_TOTAL)
                            if not OLD_LOG:
                                OLD_LOG = json_payload
                                TOTAL_OFFSET=json_payload["counts"]["total"]
                            else:
                                if OLD_LOG["direction"] == json_payload["direction"] and OLD_LOG["event_type"] == json_payload["event_type"] and OLD_LOG["counts"]["total"] == json_payload["counts"]["total"]:
                                    do_not_log = True
                                
                                
                                TOTAL_OFFSET=json_payload["counts"]["total"]-LOCAL_TOTAL
                                print("======================================================= total offset 1 _",TOTAL_OFFSET,LOCAL_TOTAL)

                                OLD_LOG = json_payload
                            if not do_not_log or TOTAL_OFFSET > 0:
                                
                                log_file.write(json.dumps(json_payload) + "\n")
                                if not json_payload["class_name"] == "forklift":
                                    LOCAL_TOTAL += 1
                                print("======================================================= is writing")
                                if TOTAL_OFFSET > 0:
                                    TOTAL_OFFSET-=1
                                
                            print("======================================================= total offset 2 _",TOTAL_OFFSET,LOCAL_TOTAL)
                            print("======================================================= After_log_line")
            except Exception as e:
                print(f"Could not write events log: {e}")
        self.last_counts = dict(counts)
    def _save_report(self, elapsed):
        report_dir = "result"
        # os.makedirs(report_dir, exist_ok=True)
        # report = {
        #     "video_input": self.config.VIDEO_INPUT_PATH,
        #     "model_path": self.config.MODEL_PATH,
        #     "start_time": self.start_timestamp,
        #     "end_time": time.time(),
        #     "processing_time_seconds": elapsed,
        #     "frames_processed": self.frame_number,
        #     "effective_frames": self.processed_frames,
        #     "counts": self.last_counts,
        #     "events": self.report_events,
        # }
        # timestamp = time.strftime("%Y%m%d_%H%M%S")
        # filename = f"bale_report_{timestamp}.json"
        # path = os.path.join(report_dir, filename)
        # with open(path, "w", encoding="utf-8") as f:
        #     json.dump(report, f, indent=2)
        # print(f"Report saved to {path}")
    def _cleanup(self, elapsed):
        print("\nCleaning up...")
        self.video_reader.stop()
        if self.video_writer:
            self.video_writer.stop()
        if self.config.SHOW_DISPLAY:
            cv2.destroyAllWindows()
        if self.executor:
            self.executor.shutdown(wait=True, cancel_futures=False)
        self._save_report(elapsed)
        print("\n" + "=" * 60)
        print(" Video processing completed!")
        print(f" Input: {self.config.VIDEO_INPUT_PATH}")
        print(f"  Total processing time: {elapsed:.1f} seconds")
        print(f" Frames processed: {self.frame_number}")
        print(f" Effective frames analysed: {self.processed_frames} "
              f"(skip: {self.config.PROCESS_EVERY_N_FRAMES})")
        print("=" * 60)
        # Event_log_temp=[]

def main():
    config = Config()
    app = BaleCountingApp(config)
    app.run()
if __name__ == "__main__":
    main()


