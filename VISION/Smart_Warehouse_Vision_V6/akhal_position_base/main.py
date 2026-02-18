import os
import json
import cv2
import math
import threading
import queue
import time, requests
from pathlib import Path
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from ultralytics import YOLO
try:
    import torch
except ImportError:
    torch = None
GLOBAL_OUTPUT = {}
OFFSET_THRESHOLD = 300
LAST_DETECTED = None
LAST_DETECTED_OFFSET = None
# Faster shutter for high-speed objects
#os.environ["ANBAR_EXPOSURE_US"] = os.environ.get("ANBAR_EXPOSURE_US", "8000")
class Config:
    """Configuration class for the counting system"""
    #MODEL_PATH = "weights-416-v2/best-416-v2.onnx"
    #MODEL_PATH = "weights/best-512_ncnn_model"# 
    MODEL_PATH =  "best_ncnn_model-yolo26-512-1-26"
    #MODEL_PATH = "weights/best-free-v2.onnx"
    #MODEL_PATH = "best-512-70_ncnn_model"
    #"D:\test-akhal\weights\best-free-v2.onnx"
    #MODEL_PATH = "best_ncnn_model_512_15_1"

    #MODEL_PATH = "best-512-70_ncnn_model"  #
    #MODEL_PATH ="weights/best-512-v2.onnx"
    #MODEL_PATH = "weights/best-free-v2.onnx"
    #MODEL_PATH ="weights/best.pt"
    #MODEL_PATH = "New/best-free-v2.onnx"
    
    DETECTION_IMAGE_SIZE = 512
    CONFIDENCE_THRESHOLD = 0.45
    MAX_DETECTIONS = 60
    VIDEO_INPUT_PATH = "picamera2"
    #VIDEO_INPUT_PATH = "video_source/pending/05.mp4"
    VIDEO_OUTPUT_PATH = ""
    VIDEO_CODEC = "mp4v"
    # ALLOWED_CLASSES = ["forklift", "sulfate", "pack", "fructose", "paper-roll", "akhal", "sude", "akd"]

    ALLOWED_CLASSES = ["forklift", "akhal"]
    EXCLUDE_CLASSES = []
    FIXED_ID_CLASSES = ["forklift"]
    MATCH_DISTANCE = 120
    MAX_MISSED_FRAMES = 30
    SHOW_DISPLAY = True
    DISPLAY_WINDOW_NAME = " Counter"
    QUEUE_SIZE = 32
    FPS_SMOOTHING_FRAMES = 30
    PROCESS_EVERY_N_FRAMES = 1
    DETECTION_PIPELINE_DEPTH = 2
    DETECTION_WORKERS = 2
    ENTRY_LINE_X =1550 # 300 #1600
    BBOX_ALPHA = 0.6
    JSON_FILE_PATH = "temp2/akhal_tracking.json"
    # LOG_FILE_PATH = "temp2/akhal_tracking_log.jsonl"
    COLOR_BBOX = (0, 255, 0)
    COLOR_COUNTED_OBJECT = (0, 255, 255)
    COLOR_FPS_TEXT = (50, 255, 255)
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
    def __init__(self, config):
        self.config = config
    def annotate_frame(self, frame, tracked_objects, counts, fps):
        annotated_frame = frame.copy()
        entry_line_x = getattr(self.config, "ENTRY_LINE_X", None)
        if entry_line_x is not None:
            cv2.line(annotated_frame, (entry_line_x, 0), (entry_line_x, annotated_frame.shape[0] - 1), (0, 0, 255), 2)
        self._draw_tracks(annotated_frame, tracked_objects)
        self._draw_info_overlay(annotated_frame, counts, fps)
        return annotated_frame
    def _draw_tracks(self, frame, tracked_objects):
        for track in tracked_objects:
            bbox = track.get("bbox")
            if bbox is None:
                continue
            color = self.config.COLOR_BBOX
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (track["cx"], track["cy"]), 5, color, -1)
            label = f'ID:{track["track_id"]}'# Conf:{track["conf"]:.2f}'
            cv2.putText(frame, label, (x1, max(y1 - 5, 0)), self.config.FONT,
                        self.config.FONT_SCALE_SMALL, color, self.config.FONT_THICKNESS)
    def _draw_info_overlay(self, frame, counts, fps):
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), self.config.FONT,
                    self.config.FONT_SCALE_MEDIUM, self.config.COLOR_FPS_TEXT, self.config.FONT_THICKNESS)
        total_count = counts.get("total", 0)
        cv2.putText(frame, f"Total: {total_count}", (10, 65), self.config.FONT,
                    self.config.FONT_SCALE_MEDIUM, self.config.COLOR_TOTAL_TEXT, self.config.FONT_THICKNESS)
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
            cx = int((x1 + x2) / 2) #x2
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
    def __init__(self, config):
        self.match_distance = config.MATCH_DISTANCE
        self.max_missed_frames = config.MAX_MISSED_FRAMES
        self.tracks = {}
        self.next_track_id = 0
        self.events = []
        self.class_states = defaultdict(dict)
        self.total_entered = defaultdict(int)
        self.total_exit = defaultdict(int)
        self.total_internal = defaultdict(int)
        self.bbox_alpha = getattr(config, "BBOX_ALPHA", 0.6)
        self.entry_line_x = getattr(config, "ENTRY_LINE_X", 1600)
        self.allowed_classes = set(getattr(config, "ALLOWED_CLASSES", []))
        self.json_file_path = getattr(config, "JSON_FILE_PATH", "temp2/akhal_tracking.json")
        self.fixed_id_classes = {cls.lower() for cls in getattr(config, "FIXED_ID_CLASSES", [])}
        self.fixed_track_ids = {}
        # self.log_file_path = getattr(config, "LOG_FILE_PATH", "temp2/akhal_tracking_log.jsonl")
    def update(self, detections, frame_index):
        self.events = []
        for track in self.tracks.values():
            track["matched"] = False
        for track in self.tracks.values():
            track["kalman"].predict()
        tracked_visuals = []
        fixed_class_used = set()
        for det in detections:
            cx, cy = det["cx"], det["cy"]
            class_name = det["class_name"]
            if class_name in self.fixed_id_classes:
                if class_name in fixed_class_used:
                    continue
                fixed_class_used.add(class_name)
                fixed_id = self.fixed_track_ids.get(class_name)
                if fixed_id is not None:
                    if fixed_id not in self.tracks:
                        self._create_track(cx, cy, class_name, frame_index, track_id=fixed_id)
                    best_id = fixed_id
                else:
                    best_id = self._create_track(cx, cy, class_name, frame_index)
                    self.fixed_track_ids[class_name] = best_id
            else:
                best_id = self._find_best_track(cx, cy, class_name)
                if best_id is None:
                    best_id = self._create_track(cx, cy, class_name, frame_index)
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
            track["class_name"] = class_name
            track["bbox"] = det["bbox"]
            track["conf"] = det["conf"]
            tracked_visuals.append({
                "track_id": best_id,
                "cx": smooth_cx,
                "cy": smooth_cy,
                "bbox": det["bbox"],
                "class_name": class_name,
                "conf": det["conf"],
            })
        class_tracks = defaultdict(list)
        for track in tracked_visuals:
            if not self.allowed_classes or track["class_name"] in self.allowed_classes:
                class_tracks[track["class_name"]].append(track)
        forklift_x1 = None
        forklift_present = False
        forklift_in_zone = False
        forklift_outside_zone = False
        forklift_states = self.class_states.get("forklift", {})
        active_forklift_x1s = []
        for track in tracked_visuals:
            if track["class_name"] != "forklift":
                continue
            forklift_present = True
            if track["cx"] < self.entry_line_x:
                forklift_in_zone = True
            else:
                forklift_outside_zone = True
            st = forklift_states.get(track["track_id"])
            if st is not None:
                active_forklift_x1s.append(st.get("x1"))
        if active_forklift_x1s:
            forklift_x1 = max(active_forklift_x1s)
        forklift_from_outside = forklift_x1 is not None and forklift_x1 > self.entry_line_x
        forklift_from_inside = forklift_x1 is not None and forklift_x1 <= self.entry_line_x
        for class_name, items in class_tracks.items():
            class_states = self.class_states[class_name]
            frame_items = []
            for track in items:
                x1, y1, x2, y2 = track["bbox"]
                per_id = track["track_id"]
                smooth_cx = float(track["cx"])
                smooth_cy = float(track["cy"])
                cur_w = max(1, int(x2 - x1))
                cur_h = max(1, int(y2 - y1))
                if per_id not in class_states:
                    smooth_w = cur_w
                    smooth_h = cur_h
                    smooth_x1 = int(smooth_cx - (smooth_w / 2))
                    smooth_y1 = int(smooth_cy - (smooth_h / 2))
                    smooth_x2 = int(smooth_cx + (smooth_w / 2))
                    smooth_y2 = int(smooth_cy + (smooth_h / 2))
                    smooth_bbox = (smooth_x1, smooth_y1, smooth_x2, smooth_y2)
                    center_x = int(smooth_cx)
                    class_states[per_id] = {
                        "x1": center_x,
                        "x2": center_x,
                        "situation": "",
                        "enter_counted": False,
                        "exit_counted": False,
                        "internal_counted": False,
                        "timestamp": int(time.time()),
                        "bbox": smooth_bbox
                    }
                    global LAST_DETECTED
                    global LAST_DETECTED_OFFSET
                    prev_timestamp = LAST_DETECTED["timestamp"] if LAST_DETECTED else None
                    new_last_ofsset = None if prev_timestamp is None else class_states[per_id]["timestamp"] - prev_timestamp
                    LAST_DETECTED_OFFSET = new_last_ofsset
                    LAST_DETECTED = {
                        "class_name": class_name,
                        "per_id": per_id,
                        "timestamp": class_states[per_id]["timestamp"],
                        "bbox": smooth_bbox,
                        "center_x": center_x,
                        "center_y": int(smooth_cy)
                    }
                else:
                    px1, py1, px2, py2 = class_states[per_id]["bbox"]
                    prev_w = max(1, int(px2 - px1))
                    prev_h = max(1, int(py2 - py1))
                    smooth_w = int(prev_w * (1 - self.bbox_alpha) + cur_w * self.bbox_alpha)
                    smooth_h = int(prev_h * (1 - self.bbox_alpha) + cur_h * self.bbox_alpha)
                    smooth_x1 = int(smooth_cx - (smooth_w / 2))
                    smooth_y1 = int(smooth_cy - (smooth_h / 2))
                    smooth_x2 = int(smooth_cx + (smooth_w / 2))
                    smooth_y2 = int(smooth_cy + (smooth_h / 2))
                    smooth_bbox = (smooth_x1, smooth_y1, smooth_x2, smooth_y2)
                    class_states[per_id]["bbox"] = smooth_bbox
                    center_x = int(smooth_cx)
                    class_states[per_id]["x2"] = center_x
                track["bbox"] = smooth_bbox
                frame_items.append({
                    "per_id": per_id,
                    "center_x": center_x,
                    "bbox": smooth_bbox
                })
            moving_left_items = []
            ############################################right to left
            for item in frame_items:
                st = class_states[item["per_id"]]
                if st["x2"] < st["x1"] and (st["x1"] - st["x2"] > 100):
                    moving_left_items.append(item)
            for item in frame_items:
                per_id = item["per_id"]
                center_x = st["x1"] #st["x2"] #x2# item["center_x"]
                st = class_states[per_id]


                if st["x2"] < st["x1"] and not st["enter_counted"] and (st["x1"] - st["x2"] > 100):# and st["x1"] > self.entry_line_x:# 100
                    has_parallel = False
                    for other in moving_left_items:
                        if other["per_id"] != per_id and abs(center_x - other["center_x"]) <= 100:# 100
                            has_parallel = True
                            break
                    if class_name == "forklift":
                        if st["x1"] > self.entry_line_x:
                            st["situation"] = "enter"
                            st["enter_counted"] = True
                            self.total_entered[class_name] += 1
                        else:
                            st["situation"] = "Internal Displacement"
                            if not st["internal_counted"]:
                                self.total_internal[class_name] += 1
                                st["internal_counted"] = True
                    else:
                        if not forklift_present or forklift_x1 is None:
                            pass
                        elif forklift_from_outside:
                            st["situation"] = "enter"
                            st["enter_counted"] = True
                            self.total_entered[class_name] += 1
                        else:
                            st["situation"] = "Internal Displacement"
                            if not st["internal_counted"]:
                                self.total_internal[class_name] += 1
                                st["internal_counted"] = True
                if st["x2"] > st["x1"] and not st["exit_counted"] and (st["x2"] - st["x1"] > 300): #100
                    if class_name == "forklift":
                        if st["x2"] > self.entry_line_x:
                            st["situation"] = "Exit"
                            if not st["exit_counted"]:
                                self.total_exit[class_name] += 1
                                st["exit_counted"] = True
                        else:
                            st["situation"] = "Internal Displacement"
                            if not st["internal_counted"]:
                                self.total_internal[class_name] += 1
                                st["internal_counted"] = True
                    else:
                        if not forklift_present or forklift_x1 is None:
                            pass
                        elif forklift_outside_zone:
                            st["situation"] = "Exit"
                            if not st["exit_counted"]:
                                self.total_exit[class_name] += 1
                                st["exit_counted"] = True
                        else:
                            st["situation"] = "Internal Displacement"
                            if not st["internal_counted"]:
                                self.total_internal[class_name] += 1
                                st["internal_counted"] = True
            ###########################################  right to left

            #######################  left to right
                
            #     if st["x2"] < st["x1"] and not st["exit_counted"] and (st["x1"] - st["x2"] > 100):#100
            #         has_parallel = False
            #         for other in moving_left_items:
            #             if other["per_id"] != per_id and abs(center_x - other["center_x"]) <= 100:
            #                 has_parallel = True
            #                 break
            #         if st["x1"] > self.entry_line_x or has_parallel:
            #             st["situation"] = "exit"
            #             st["exit_counted"] = True
            #             self.total_exit[class_name] += 1
            #         else:
            #             st["situation"] = "Internal Displacement"
            #             if not st["internal_counted"]:
            #                 self.total_internal[class_name] += 1
            #                 st["internal_counted"] = True
                
            #     if st["x2"] > st["x1"] and not st["enter_counted"] and (st["x2"] - st["x1"] > 100):#100
            #         if st["x2"] >  self.entry_line_x: #1570:
            #             st["situation"] = "enter"
            #             if not st["enter_counted"]:
            #                 self.total_entered[class_name] += 1
            #                 st["enter_counted"] = True
            #         else:
            #             st["situation"] = "Internal Displacement"
            #             if not st["internal_counted"]:
            #                 self.total_internal[class_name] += 1
            #                 st["internal_counted"] = True


            # ################################################



        self._purge_stale_tracks()
        total_count = self.total_entered.get("akhal", 0)
        self._write_output(frame_index)
        return {
            "tracks": tracked_visuals,
            "counts": {
                "total": total_count,
            },
            "events": list(self.events),
        }
    def _write_output(self, frame_index):
        global GLOBAL_OUTPUT
        global LAST_DETECTED
        global LAST_DETECTED_OFFSET
        output_dict = {}
        for class_name, items in self.class_states.items():
            for k, v in items.items():
                out_item = {**v, "x1": int(v["x1"]), "x2": int(v["x2"])}
                if "bbox" in out_item:
                    bx1, by1, bx2, by2 = out_item["bbox"]
                    out_item["bbox"] = [int(bx1), int(by1), int(bx2), int(by2)]
                output_dict[f"{class_name}_{k}"] = out_item
        output_dict["total_entered"] = {name: int(cnt) for name, cnt in self.total_entered.items()}
        output_dict["total"] = int(self.total_entered.get("akhal", 0))#akhal
        output_dict["total_exit"] = {name: int(cnt) for name, cnt in self.total_exit.items()}
        output_dict["total_internal"] = {name: int(cnt) for name, cnt in self.total_internal.items()}
        GLOBAL_OUTPUT = output_dict
        if LAST_DETECTED is not None and time.time() - LAST_DETECTED["timestamp"] > OFFSET_THRESHOLD :
            print("============================= SENT DATA AND CLEAR =============================\n")
            requests.post(f"http://192.168.2.21:6008/receive_data/", json=GLOBAL_OUTPUT)
            print("============================= END =============================\n")
            GLOBAL_OUTPUT = {}
            LAST_DETECTED = None
            LAST_DETECTED_OFFSET = None
            self.tracks = {}
            self.class_states = defaultdict(dict)
            self.events = []
            self.total_entered = defaultdict(int)
            self.total_exit = defaultdict(int)
            self.total_internal = defaultdict(int)
            self.fixed_track_ids = {}
            self.next_track_id = 0
    def _find_best_track(self, cx, cy, class_name):
        best_id = None
        best_distance = self.match_distance
        for track_id, track in self.tracks.items():
            if track.get("matched"):
                continue
            if track.get("class_name") != class_name:
                continue
            pred_x, pred_y = track["kalman"].get_predicted_position()
            distance = math.hypot(cx - pred_x, cy - pred_y)
            if distance < best_distance:
                best_distance = distance
                best_id = track_id
        return best_id
    def _create_track(self, cx, cy, class_name, frame_index, track_id=None):
        if track_id is None:
            self.next_track_id += 1
            track_id = self.next_track_id
        else:
            self.next_track_id = max(self.next_track_id, track_id)
        kf = KalmanFilter(cx, cy)
        self.tracks[track_id] = {
            "prev_x": cx, "prev_y": cy,
            "prev_x_prev": cx, "prev_y_prev": cy,
            "class_name": class_name,
            "matched": True,
            "missed": 0,
            "last_seen": frame_index,
            "bbox": None,
            "conf": 0.0,
            "kalman": kf,
        }
        return track_id

    def _purge_stale_tracks(self):
        stale_ids = [tid for tid, t in self.tracks.items() if not t["matched"]]
        for tid in stale_ids:
            track = self.tracks[tid]
            track["missed"] += 1
            if track["missed"] > self.max_missed_frames:
                class_name = track.get("class_name")
                if class_name in self.fixed_id_classes:
                    fixed_id = self.fixed_track_ids.get(class_name)
                    if fixed_id == tid:
                        del self.fixed_track_ids[class_name]
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

def run_simple():
    config = Config()
    video_reader = VideoReader(config.VIDEO_INPUT_PATH, config.QUEUE_SIZE)
    detector = ObjectDetector(config)
    tracker = ObjectTracker(config)
    annotator = FrameAnnotator(config)
    fps_calculator = FPSCalculator(config.FPS_SMOOTHING_FRAMES)
    video_reader.start()
    try:
        while video_reader.more():
            frame_data = video_reader.read()
            if frame_data is None:
                continue
            frame = frame_data["frame"]
            frame_number = frame_data["frame_number"]
            fps_calculator.end_frame()
            if frame_number % config.PROCESS_EVERY_N_FRAMES != 0:
                if config.SHOW_DISPLAY:
                    display = cv2.resize(frame, (1280, 720))#(640, 360)
                    cv2.imshow(config.DISPLAY_WINDOW_NAME, display)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue
            detections = detector.detect(frame)
            tracking_info = tracker.update(detections, frame_number)
            fps_value = fps_calculator.get_current_fps()
            annotated = annotator.annotate_frame(frame, tracking_info["tracks"], tracking_info["counts"], fps_value)
            if config.SHOW_DISPLAY:
                display = cv2.resize(annotated, (1280, 720))#(640, 360)
                cv2.imshow(config.DISPLAY_WINDOW_NAME, display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        video_reader.stop()
        if config.SHOW_DISPLAY:
            cv2.destroyAllWindows()

def main():
    run_simple()
if __name__ == "__main__":
    main()


