import os
import json
import logging
import cv2
import math
import threading
import queue
import time
from pathlib import Path
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
from ultralytics import YOLO

try:
    import torch
except ImportError:
    torch = None

try:
    import ncnn
except ImportError:
    ncnn = None

# Faster shutter for high-speed objects (40 km/h). Default 1/125s (8000 µs).
os.environ["ANBAR_EXPOSURE_US"] = os.environ.get("ANBAR_EXPOSURE_US", "8000")


class Config:
    """Configuration class for the counting system"""
    # MODEL_PATH ="best_50_new_ncnn_model_with_auggmentation_v1"
    #MODEL_PATH ="weights/best_50_v1.onnx"
    # MODEL_PATH ="best-512_ncnn_model"
    MODEL_PATH ="weights-512-70/best-512-70_ncnn_model"


    # MODEL_PATH ="best_70_new_ncnn_model_with_auggmentation"
    # MODEL_PATH = "best-50-new-yolov11_ncnn_model"
    DETECTION_IMAGE_SIZE = 512
    CONFIDENCE_THRESHOLD = 0.2
    MAX_DETECTIONS = 30

    # VIDEO_INPUT_PATH = "video_source/pending/13.mp4"
    VIDEO_INPUT_PATH = "video_source/pending/14.mp4"

    VIDEO_OUTPUT_PATH = ""
    VIDEO_CODEC = "mp4v"

    PRIMARY_LINE_POSITION = 4 / 6
    SECONDARY_LINE_POSITION = 1 / 6
    COUNTING_ZONE_MARGIN = 45

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

    FONT = 0
    FONT_SCALE_LARGE = 0.9
    FONT_SCALE_MEDIUM = 0.7
    FONT_SCALE_SMALL = 0.5
    FONT_THICKNESS = 2

    VERBOSE = True


def apply_performance_settings():
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"

    cv2.setNumThreads(4)
    cv2.setUseOptimized(True)

    if torch is not None:
        torch.set_num_threads(4)


def configure_runtime():
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
    cv2.setNumThreads(4)
    cv2.setUseOptimized(True)
    if torch is not None:
        torch.set_num_threads(4)


print("fps start")


class FPSCalculator:
    """FPS calculator with smoothing"""

    def __init__(self, smoothing_frames=30):
        self.smoothing_frames = smoothing_frames
        self.timestamps = deque(maxlen=smoothing_frames)

    def start_frame(self):
        return

    def end_frame(self):
        now = time.time()
        self.timestamps.append(now)

        if len(self.timestamps) >= 2:
            duration = self.timestamps[-1] - self.timestamps[0]
            if duration > 0:
                return (len(self.timestamps) - 1) / duration

        return 0.0

    def get_current_fps(self):
        if len(self.timestamps) >= 2:
            duration = self.timestamps[-1] - self.timestamps[0]
            if duration > 0:
                return (len(self.timestamps) - 1) / duration
        return 0.0


print("frame-annotation start")


class FrameAnnotator:
    """Frame annotation and visualization"""

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
            cv2.circle(frame, (track["cx"], track["cy"]), 3, color, -1)
            label = f'ID:{track["track_id"]} Conf:{track["conf"]:.2f}'
            text_origin = (x1, max(y1 - 5, 0))
            cv2.putText(
                frame,
                label,
                text_origin,
                self.config.FONT,
                self.config.FONT_SCALE_SMALL,
                color,
                self.config.FONT_THICKNESS,
            )

    def _draw_counting_lines(self, frame):
        height = frame.shape[0]

        cv2.line(
            frame,
            (self.primary_line_x, 0),
            (self.primary_line_x, height),
            self.config.COLOR_COUNTING_PRIMARY,
            3,
        )
        cv2.line(
            frame,
            (self.primary_line_x - self.margin, 0),
            (self.primary_line_x - self.margin, height),
            self.config.COLOR_ZONE_BOUNDARY_PRIMARY,
            1,
        )
        cv2.line(
            frame,
            (self.primary_line_x + self.margin, 0),
            (self.primary_line_x + self.margin, height),
            self.config.COLOR_ZONE_BOUNDARY_PRIMARY,
            1,
        )

        cv2.line(
            frame,
            (self.secondary_line_x, 0),
            (self.secondary_line_x, height),
            self.config.COLOR_COUNTING_SECONDARY,
            3,
        )
        cv2.line(
            frame,
            (self.secondary_line_x - self.margin, 0),
            (self.secondary_line_x - self.margin, height),
            self.config.COLOR_ZONE_BOUNDARY_SECONDARY,
            1,
        )
        cv2.line(
            frame,
            (self.secondary_line_x + self.margin, 0),
            (self.secondary_line_x + self.margin, height),
            self.config.COLOR_ZONE_BOUNDARY_SECONDARY,
            1,
        )

    def _draw_info_overlay(self, frame, counts, fps):
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            self.config.FONT,
            self.config.FONT_SCALE_MEDIUM,
            self.config.COLOR_FPS_TEXT,
            self.config.FONT_THICKNESS,
        )
        cv2.putText(
            frame,
            f'Primary: {counts["primary"]}',
            (10, 65),
            self.config.FONT,
            self.config.FONT_SCALE_MEDIUM,
            self.config.COLOR_PRIMARY_TEXT,
            self.config.FONT_THICKNESS,
        )
        cv2.putText(
            frame,
            f'Secondary: {counts["secondary"]}',
            (10, 100),
            self.config.FONT,
            self.config.FONT_SCALE_MEDIUM,
            self.config.COLOR_SECONDARY_TEXT,
            self.config.FONT_THICKNESS,
        )
        cv2.putText(
            frame,
            f'Total: {counts["total"]}',
            (10, 135),
            self.config.FONT,
            self.config.FONT_SCALE_MEDIUM,
            self.config.COLOR_TOTAL_TEXT,
            self.config.FONT_THICKNESS,
        )

        live_count = counts.get("live", 1)
        cv2.putText(
            frame,
            f"Live: {live_count}",
            (10, 170),
            self.config.FONT,
            self.config.FONT_SCALE_MEDIUM,
            self.config.COLOR_TOTAL_TEXT,
            self.config.FONT_THICKNESS,
        )

        last_obj = counts.get("last_non_forklift_primary_class")
        if last_obj:
            cv2.putText(
                frame,
                f"Last: {last_obj}",
                (10, 205),
                self.config.FONT,
                self.config.FONT_SCALE_MEDIUM,
                self.config.COLOR_PRIMARY_TEXT,
                self.config.FONT_THICKNESS,
            )

        forklift_entry = counts.get("forklift_entry", 0)
        forklift_exit = counts.get("forklift_exit", 0)

        cv2.putText(
            frame,
            f"Forklift In: {forklift_entry}",
            (10, 240),
            self.config.FONT,
            self.config.FONT_SCALE_MEDIUM,
            self.config.COLOR_PRIMARY_TEXT,
            self.config.FONT_THICKNESS,
        )
        cv2.putText(
            frame,
            f"Forklift Out: {forklift_exit}",
            (10, 275),
            self.config.FONT,
            self.config.FONT_SCALE_MEDIUM,
            self.config.COLOR_SECONDARY_TEXT,
            self.config.FONT_THICKNESS,
        )


print("object-detector-start")


class ObjectDetector:
    """YOLO-based object detector tailored for bale counting"""

    def __init__(self, config):
        self.imgsz = config.DETECTION_IMAGE_SIZE
        self.conf_threshold = config.CONFIDENCE_THRESHOLD
        self.max_detections = getattr(config, "MAX_DETECTIONS", 30)
        self.exclude_classes = [cls.lower() for cls in config.EXCLUDE_CLASSES]
        self.model_path = Path(config.MODEL_PATH)
        self.uses_ncnn = self._is_ncnn_model(self.model_path)
        self.model = YOLO(str(self.model_path),task="detect")
        self.device = None
        self.use_torch_context = torch is not None and not self.uses_ncnn
        if self.use_torch_context:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)

    def detect(self, frame):
        predict_kwargs = {
            "verbose": False,
            "imgsz": self.imgsz,
            "conf": self.conf_threshold,
            "max_det": self.max_detections,
        }

        detections = []
        if self.use_torch_context:
            with torch.inference_mode():
                results = self.model(frame, **predict_kwargs)
        else:
            results = self.model(frame, **predict_kwargs)

        if not results:
            return detections

        result = results[0]
        if hasattr(result, "obb") and result.obb is not None and len(result.obb) > 0:
            detections.extend(self._parse_obb_detections(result))
            return detections

        if hasattr(result, "boxes") and result.boxes is not None and len(result.boxes) > 0:
            detections.extend(self._parse_box_detections(result))

        return detections

    def _is_ncnn_model(self, path: Path):
        if path.is_dir():
            return any(path.glob("*.ncnn.param"))
        suffix = path.suffix.lower()
        return suffix in {".ncnn", ".param", ".bin"}

    def _to_numpy(self, value):
        data = value
        if hasattr(data, "cpu"):
            data = data.cpu()
        if hasattr(data, "numpy"):
            return data.numpy()
        return np.asarray(data)

    def _parse_obb_detections(self, result):
        detections = []
        obb_data = result.obb
        obb_boxes = self._to_numpy(obb_data.xyxyxyxy)
        classes = self._to_numpy(obb_data.cls).astype(int)
        confidences = self._to_numpy(obb_data.conf)
        xywhr = self._to_numpy(obb_data.xywhr)

        for obb_box, cls_id, conf, xywhr_data in zip(obb_boxes, classes, confidences, xywhr):
            class_name = self.model.names[int(cls_id)].lower()
            if class_name in self.exclude_classes:
                continue

            cx = int(xywhr_data[0])
            cy = int(xywhr_data[1])
            points = obb_box.reshape((-1, 2))
            x_min = int(points[:, 0].min())
            y_min = int(points[:, 1].min())
            x_max = int(points[:, 0].max())
            y_max = int(points[:, 1].max())

            detections.append(
                {
                    "cx": cx,
                    "cy": cy,
                    "bbox": (x_min, y_min, x_max, y_max),
                    "class_name": class_name,
                    "conf": float(conf),
                }
            )

        return detections

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

            detections.append(
                {
                    "cx": cx,
                    "cy": cy,
                    "bbox": (x1, y1, x2, y2),
                    "class_name": class_name,
                    "conf": float(conf),
                }
            )

        return detections


print("object-tracker-start")


class ObjectTracker:
    """Centroid-based tracker with dual counting lines and class-specific counters"""

    def __init__(self, primary_line_x, secondary_line_x, config):
        self.primary_line_x = primary_line_x
        self.secondary_line_x = secondary_line_x
        self.counting_zone_margin = config.COUNTING_ZONE_MARGIN
        self.match_distance = config.MATCH_DISTANCE
        self.max_missed_frames = config.MAX_MISSED_FRAMES

        self.forklift_class_name = getattr(config, "FORKLIFT_CLASS_NAME", "forklift").lower()
        self.bale_class_name = getattr(config, "BALE_CLASS_NAME", "bale").lower()

        self.tracks = {}
        self.next_track_id = 0

        self.primary_count = 0
        self.secondary_count = 0
        self.total_crossings = 0

        self.forklift_entry_count = False
        self.forklift_exit_count = False

        self.bale_entry_count = 0
        self.bale_exit_count = 0

        self.last_non_forklift_primary_class = ""

        self.forklift_presence = 0
        self.current_live_bales = 0
        self.pending_live_bales = 0

        self.events = []

    def update(self, detections, frame_index):
        self.events = []

        for track in self.tracks.values():
            track["matched"] = False

        tracked_visuals = []
        self.current_live_bales = 0
        for det in detections:
            cx, cy = det["cx"], det["cy"]
            best_id = self._find_best_track(cx, cy)

            if best_id is None:
                best_id = self._create_track(cx, cy, det["class_name"], frame_index)

            track = self.tracks[best_id]
            track["matched"] = True
            track["missed"] = 0
            track["last_seen"] = frame_index
            track["prev_x_prev"] = track["prev_x"]
            track["prev_y_prev"] = track["prev_y"]
            track["prev_x"] = cx
            track["prev_y"] = cy
            track["class_name"] = det["class_name"]
            track["bbox"] = det["bbox"]
            track["conf"] = det["conf"]
            self._check_crossings(best_id, track)

            tracked_visuals.append(
                {
                    "track_id": best_id,
                    "cx": cx,
                    "cy": cy,
                    "bbox": det["bbox"],
                    "class_name": det["class_name"],
                    "conf": det["conf"],
                    "counted_primary": track["counted_primary"],
                    "counted_secondary": track["counted_secondary"],
                }
            )

        self._purge_stale_tracks()

        return {
            "tracks": tracked_visuals,
            "counts": {
                "primary": self.primary_count,
                "secondary": self.secondary_count,
                "total": max(self.primary_count, self.secondary_count),
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
            distance = math.hypot(cx - track["prev_x"], cy - track["prev_y"])
            if distance < best_distance:
                best_distance = distance
                best_id = track_id

        return best_id

    def _create_track(self, cx, cy, class_name, frame_index):
        self.next_track_id += 1

        class_name_lc = str(class_name).lower()
        is_forklift = class_name_lc == self.forklift_class_name

        if is_forklift:
            counted_primary = False
            counted_secondary = False
            forklift_entry_primary_done = False
            forklift_exit_primary_done = False
            forklift_entry_secondary_done = False
        else:
            counted_primary = cx <= self.primary_line_x
            counted_secondary = cx <= self.secondary_line_x
            forklift_entry_primary_done = False
            forklift_exit_primary_done = False
            forklift_entry_secondary_done = False

        self.tracks[self.next_track_id] = {
            "prev_x": cx,
            "prev_y": cy,
            "prev_x_prev": cx,
            "prev_y_prev": cy,
            "class_name": class_name,
            "counted_primary": counted_primary,
            "counted_secondary": counted_secondary,
            "forklift_entry_primary_done": forklift_entry_primary_done,
            "forklift_exit_primary_done": forklift_exit_primary_done,
            "forklift_entry_secondary_done": forklift_entry_secondary_done,
            "matched": True,
            "missed": 0,
            "last_seen": frame_index,
            "bbox": None,
            "conf": 0.0,
        }
        return self.next_track_id

    def _check_crossings(self, track_id, track):
        prev_prev_x = track["prev_x_prev"]
        prev_x = track["prev_x"]

        if prev_prev_x is None:
            return

        class_name = str(track["class_name"]).lower()
        is_forklift = class_name == self.forklift_class_name
        is_bale = class_name == self.bale_class_name

        primary_rtl = prev_prev_x > self.primary_line_x and prev_x <= self.primary_line_x
        primary_ltr = prev_prev_x < self.primary_line_x and prev_x >= self.primary_line_x

        secondary_rtl = prev_prev_x > self.secondary_line_x and prev_x <= self.secondary_line_x
        secondary_ltr = prev_prev_x < self.secondary_line_x and prev_x >= self.secondary_line_x
        self.pending_live_bales = 0
        if is_forklift:
            if (
                primary_rtl
                and abs(prev_x - self.primary_line_x) <= self.counting_zone_margin
                and not track.get("forklift_entry_primary_done", False)
            ):
                self.forklift_exit_count = False
                self.forklift_entry_count = True
                track["forklift_entry_primary_done"] = True
                track["counted_primary"] = True
                previous_presence = self.forklift_presence

                self.forklift_presence += 1
                if previous_presence == 0:
                    self.current_live_bales = 1
                else:
                    self.current_live_bales = max(self.current_live_bales, 1)

                self.events.append(
                    {
                        "track_id": track_id,
                        "frame_number": track.get("last_seen"),
                        "class_name": track["class_name"],
                        "line": "primary",
                        "direction": "right_to_left",
                        "event_type": "forklift_entry",
                        "live_count": self.current_live_bales,
                    }
                )
            # else:
                # self.forklift_exit_count = False
                # self.forklift_entry_count = True

            if (
                primary_ltr
                and abs(prev_x - self.primary_line_x) <= self.counting_zone_margin
                and not track.get("forklift_exit_primary_done", False)
            ):
                track["forklift_exit_primary_done"] = True
                track["counted_primary"] = True
                self.forklift_exit_count = True
                self.forklift_entry_count = False
                self.forklift_presence = max(0, self.forklift_presence - 1)

                self.current_live_bales = max(self.current_live_bales, 1)

                self.events.append(
                    {
                        "track_id": track_id,
                        "frame_number": track.get("last_seen"),
                        "class_name": track["class_name"],
                        "line": "primary",
                        "direction": "left_to_right",
                        "event_type": "forklift_exit",
                        "live_count": self.current_live_bales,
                    }
                )
            # else:
            #     self.forklift_exit_count = False
            #     self.forklift_entry_count = True
        else:
            crossed_primary = (
                not track["counted_primary"]
                and primary_rtl
                and abs(prev_x - self.primary_line_x) <= self.counting_zone_margin
            )

            if crossed_primary:
                self.forklift_entry_count = True
                self.forklift_exit_count = False
                self.primary_count += 1
                self.total_crossings += 1
                if is_bale:
                    self.bale_entry_count += 1
                    event_type = "bale_entry"
                    self.pending_live_bales += 1
                else:
                    event_type = "other_entry"

                self.last_non_forklift_primary_class = str(track["class_name"])

                self.current_live_bales = max(self.current_live_bales, self.pending_live_bales)

                self.events.append(
                    {
                        "track_id": track_id,
                        "frame_number": track.get("last_seen"),
                        "class_name": track["class_name"],
                        "line": "primary",
                        "direction": "right_to_left",
                        "event_type": event_type,
                    }
                )
            # else:
            #     self.forklift_entry_count = False
            #     self.forklift_exit_count = True

        if is_forklift:
            pass
        else:
            crossed_secondary = (
                not track["counted_secondary"]
                and secondary_rtl
                and abs(prev_x - self.secondary_line_x) <= self.counting_zone_margin
            )

            if crossed_secondary:
                track["counted_secondary"] = True

                self.secondary_count += 1
                self.total_crossings += 1
                if is_bale:
                    self.bale_exit_count += 1
                    event_type = "bale_exit"
                    self.pending_live_bales += 1
                else:
                    event_type = "other_exit"

                self.events.append(
                    {
                        "track_id": track_id,
                        "frame_number": track.get("last_seen"),
                        "class_name": track["class_name"],
                        "line": "secondary",
                        "direction": "right_to_left",
                        "event_type": event_type,
                    }
                )

    def _purge_stale_tracks(self):
        stale_ids = [track_id for track_id, track in self.tracks.items() if not track["matched"]]

        for track_id in stale_ids:
            track = self.tracks[track_id]
            track["missed"] += 1
            if track["missed"] > self.max_missed_frames:
                del self.tracks[track_id]


print("video-reader-start")

try:
    from picamera2 import Picamera2

    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False


class VideoReader:
    """Thread-safe video reader class"""

    def __init__(self, video_path, queue_size=128):
        self.video_path = video_path
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.thread = None

        self.cap = None
        self.picam2 = None

        if video_path == "picamera2" and PICAMERA2_AVAILABLE:
            self._init_picamera2()
        else:
            self._init_opencv_capture(video_path)

    def _init_opencv_capture(self, source):
        if isinstance(source, str) and not source.isdigit() and not os.path.exists(source):
            if PICAMERA2_AVAILABLE:
                print(f"{source} not found, switching to PiCamera2")
                self._init_picamera2()
                return

        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            if PICAMERA2_AVAILABLE:
                print(f"Cannot open {source}, switching to PiCamera2")
                self.cap.release()
                self.cap = None
                self._init_picamera2()
                return
            raise ValueError(f"Cannot open video file or camera: {source}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def _init_picamera2(self):
        try:
            tuning = Picamera2.load_tuning_file("imx219_noir.json")
            self.picam2 = Picamera2(tuning=tuning)
            config = self.picam2.create_preview_configuration(
                main={"size": (1920, 1080), "format": "RGB888"},
            )
            config["raw"]["size"] = (1920, 1080)
            self.picam2.configure(config)
            self.picam2.start()
            time.sleep(2)

            self.width = config["main"]["size"][0]
            self.height = config["main"]["size"][1]
            self.fps = 30.0
            self.total_frames = 0
        except Exception as e:
            print(f"PiCamera2 initialization failed: {e}")
            self.picam2 = None
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


print("wideo-writer-start")


class VideoWriter:
    """Thread-safe video writer class"""

    def __init__(self, output_path, fps, frame_size, codec="mp4v", queue_size=128):
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size

        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

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
                print("Warning: Video writer queue is full, dropping frame")

    def stop(self):
        self.stopped = True
        if self.thread is not None:
            self.thread.join()

    def get_frames_written(self):
        return self.frames_written


def test_inference():
    if ncnn is None:
        raise ImportError("ncnn is required for test_inference")
    if torch is None:
        raise ImportError("torch is required for test_inference")

    torch.manual_seed(0)
    in0 = torch.rand(1, 3, 640, 640, dtype=torch.float)
    out = []

    with ncnn.Net() as net:
        net.load_param("best-512_ncnn_model/model.ncnn.param")
        net.load_model("best-512_ncnn_model/model.ncnn.bin")

        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(in0.squeeze(0).numpy()).clone())

            _, out0 = ex.extract("out0")
            out.append(torch.from_numpy(np.array(out0)).unsqueeze(0))

    if len(out) == 1:
        return out[0]
    else:
        return tuple(out)

apply_performance_settings()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("log.txt", mode="a", encoding="utf-8"),
    ],
)


class BaleCountingApp:
    """Coordinator that wires together modular components"""
    
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
                config.VIDEO_OUTPUT_PATH,
                props["fps"],
                (props["width"], props["height"]),
                codec=config.VIDEO_CODEC,
                queue_size=config.QUEUE_SIZE,
            ).start()
        
        self.processed_frames = 0
        self.total_frames = props["total_frames"]
        self.frame_number = 0

        # External stop / status control
        self._stop_requested = False
        self._is_running = False

        # Reporting state
        self.report_events = []
        # Track last counts for change detection (includes class-specific counters)
        self.last_counts = {
            "primary": 0,
            "secondary": 0,
            "total": 0,
            "forklift_entry": False,
            "forklift_exit": False,
            "bale_entry": 0,
            "bale_exit": 0,
            "live": 0,
        }
        self.start_timestamp = None

        logging.info(
            "BaleCountingApp initialized input=%s model=%s width=%s height=%s fps=%s",
            self.config.VIDEO_INPUT_PATH,
            self.config.MODEL_PATH,
            props["width"],
            props["height"],
            props["fps"],
        )

    def request_stop(self):
        """Signal the app to stop processing as soon as possible."""
        self._stop_requested = True

    def run(self):
        print(" Starting bale counting (modular mode)")
        print(f"   Input: {self.config.VIDEO_INPUT_PATH}")
        print(f"   Model: {self.config.MODEL_PATH}")
        print(f"   Frame skip: {self.config.PROCESS_EVERY_N_FRAMES}")
        start_time = time.time()
        self.start_timestamp = datetime.now().isoformat()

        logging.info("Run started")
        
        # Reset flags on each fresh run
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
                
                self.fps_calculator.start_frame()
                try:
                    detections = future.result()
                except Exception as exc:
                    self.fps_calculator.end_frame()
                    print(f"\ Detection error on frame {self.frame_number}: {exc}")
                    logging.error(
                        "Detection error on frame %s: %s",
                        self.frame_number,
                        exc,
                    )
                    continue

                tracking_info = self.tracker.update(detections, self.frame_number)
                
                fps_value = self.fps_calculator.end_frame()
                annotated = self.annotator.annotate_frame(
                    frame,
                    tracking_info["tracks"],
                    tracking_info["counts"],
                    fps_value,
                )
                
                # Reporting, snapshots, and real-time JSON events when counts change
                self._handle_reporting(
                    annotated,
                    tracking_info["counts"],
                    tracking_info.get("events", []),
                )
                
                if self.video_writer:
                    self.video_writer.write(annotated)
                
                if self.config.SHOW_DISPLAY:
                    self._display_frame(annotated)
                
                self.processed_frames += 1
                if self.processed_frames % 30 == 0:
                    self._print_progress(tracking_info["counts"])
        except KeyboardInterrupt:
            print("\n Interrupted by user")
        finally:
            elapsed = time.time() - start_time
            self._cleanup(elapsed)
            # Mark as not running after cleanup
            self._is_running = False
    
    def _display_frame(self, frame):
        display_frame = cv2.resize(frame, (960, 540))
        cv2.imshow(self.config.DISPLAY_WINDOW_NAME, display_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise KeyboardInterrupt
    
    def _print_progress(self, counts):
        progress = (
            (self.frame_number / self.total_frames) * 100
            if self.total_frames > 0
            else 0
        )
        print(
            f" {self.frame_number}/{self.total_frames} ({progress:.1f}%) | "
            f"Processed: {self.processed_frames} | "
            f"Primary: {counts['primary']} | Secondary: {counts['secondary']}"
        )
    
    def _cleanup(self, elapsed):
        print("\n Cleaning up...")
        self.video_reader.stop()
        if self.video_writer:
            self.video_writer.stop()
        if self.config.SHOW_DISPLAY:
            cv2.destroyAllWindows()
        if self.executor:
            self.executor.shutdown(wait=True, cancel_futures=False)

        # Save JSON report at the end
        self._save_report(elapsed)

        logging.info(
            "Run finished elapsed=%.1fs frames=%s primary=%s secondary=%s total=%s",
            elapsed,
            self.frame_number,
            self.last_counts["primary"],
            self.last_counts["secondary"],
            self.last_counts["total"],
        )
        
        print("\n" + "=" * 60)
        print(" Video processing completed!")
        print(f" Input: {self.config.VIDEO_INPUT_PATH}")
        print(f"  Total processing time: {elapsed:.1f} seconds")
        print(f" Frames processed: {self.frame_number}")
        print(
            f" Effective frames analysed: {self.processed_frames} "
            f"(skip: {self.config.PROCESS_EVERY_N_FRAMES})"
        )
        print("=" * 60)

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
        report_dir = "result"
        temp_dir = "temp"
        os.makedirs(report_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)
        
        # Detect any increase in bale/global counters for snapshot/reporting
        increased = (
            counts["primary"] > self.last_counts.get("primary", 0)
            or counts["secondary"] > self.last_counts.get("secondary", 0)
            or counts["total"] > self.last_counts.get("total", 0)
        )
        
        snapshot_path = None
        if increased:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"snapshot_{self.frame_number}_{counts['total']}_{timestamp}.jpg"
            snapshot_path = os.path.join(report_dir, filename)
            cv2.imwrite(snapshot_path, frame)
            
            self.report_events.append({
                "frame_number": self.frame_number,
                "timestamp": datetime.now().isoformat(),
                "counts": dict(counts),
                "snapshot_path": snapshot_path,
                "live": counts.get("live", 0),
            })
            
            logging.info(
                "Count increased frame=%s primary=%s secondary=%s total=%s snapshot=%s",
                self.frame_number,
                counts["primary"],
                counts["secondary"],
                counts["total"],
                snapshot_path,
            )

        # Real-time JSON events for each forklift/bale counting event.
        # Instead of many small files, append all events to a single log file
        # in the temp directory (newline-delimited JSON).
        events_log_path = os.path.join(temp_dir, "events_log.json")
        if events:
            try:
                with open(events_log_path, "a", encoding="utf-8") as log_file:
                    for event in events:
                        event_type = event.get("event_type") or ""
                        class_name = str(event.get("class_name", "")).lower()

                        # Only emit JSON for forklift or bale events
                        if not (
                            event_type.startswith("forklift_")
                            or event_type.startswith("bale_")
                        ):
                            continue

                        # Per-event snapshot with timestamp in filename; sorting
                        # by name descending will show the newest snapshots first.
                        snapshot_ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        event_frame_number = event.get("frame_number", self.frame_number)
                        event_snapshot_path = os.path.join(
                            report_dir,
                            f"snapshot_{snapshot_ts}_{event_type}_{event_frame_number}.jpg",
                        )
                        cv2.imwrite(event_snapshot_path, frame)

                        event_timestamp = datetime.now().isoformat()
                        json_payload = {
                            "timestamp": event_timestamp,
                            "frame_number": event.get("frame_number", self.frame_number),
                            "track_id": event.get("track_id"),
                            "class_name": class_name,
                            "line": event.get("line"),
                            "direction": event.get("direction"),
                            "event_type": event_type,
                            # Snapshot for this specific event
                            "snapshot_path": event_snapshot_path,
                            # Full counter state at this moment
                            "counts": dict(counts),
                            "live": counts.get("live", 1),
                            "device_id": "anbar_akhal",
                        }

                        if json_payload["counts"].get("live", 0) != 0:
                            log_file.write(json.dumps(json_payload) + "\n")

                        logging.info(
                            "Real-time event JSON logged type=%s frame=%s path=%s",
                            event_type,
                            json_payload["frame_number"],
                            events_log_path,
                        )
            except PermissionError as e:
                # If the log file is open/locked by another program (e.g. Excel),
                # avoid crashing the main loop and just log a warning.
                logging.warning(
                    "Could not write events log file %s: %s", events_log_path, e
                )

        # Update last counts after handling this frame
        self.last_counts = dict(counts)

    def _save_report(self, elapsed):
        report_dir = "result"
        os.makedirs(report_dir, exist_ok=True)

        report = {
            "video_input": self.config.VIDEO_INPUT_PATH,
            "model_path": self.config.MODEL_PATH,
            "start_time": self.start_timestamp,
            "end_time": datetime.now().isoformat(),
            "processing_time_seconds": elapsed,
            "frames_processed": self.frame_number,
            "effective_frames": self.processed_frames,
            "counts": self.last_counts,
            "events": self.report_events,
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"bale_report_{timestamp}.json"
        path = os.path.join(report_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print(f"Report saved to {path}")
        logging.info("Report saved to %s", path)


def main():
    config = Config()
    app = BaleCountingApp(config)
    app.run()


# Simple API for starting/stopping from other modules (e.g. UI, service)
_GLOBAL_APP_INSTANCE = None


def start_app():
    """
    Create and start the BaleCountingApp.

    Returns the app instance so callers can inspect state if needed.
    """
    global _GLOBAL_APP_INSTANCE
    if _GLOBAL_APP_INSTANCE is not None:
        return _GLOBAL_APP_INSTANCE

    config = Config()
    _GLOBAL_APP_INSTANCE = BaleCountingApp(config)
    _GLOBAL_APP_INSTANCE.run()
    return _GLOBAL_APP_INSTANCE


def stop_app():
    """
    Request a graceful stop of the running BaleCountingApp.
    """
    global _GLOBAL_APP_INSTANCE
    if _GLOBAL_APP_INSTANCE is None:
        return
    _GLOBAL_APP_INSTANCE.request_stop()


def status_app():
    """
    Return current status of the BaleCountingApp as a simple string:

    - "start": app instance exists and is currently running
    - "stop":  no instance is running (never started, finished, or stop requested)
    """
    global _GLOBAL_APP_INSTANCE
    if _GLOBAL_APP_INSTANCE is None:
        return "stop"

    app = _GLOBAL_APP_INSTANCE
    # Running if loop is active and no stop has been requested
    if getattr(app, "_is_running", False) and not getattr(app, "_stop_requested", False):
        return "start"

    return "stop"


if __name__ == "__main__":
    main()

