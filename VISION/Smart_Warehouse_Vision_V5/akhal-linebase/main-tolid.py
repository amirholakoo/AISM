# from re import T
from ultralytics import YOLO
import cv2
import json
from datetime import datetime
import os
import time
import numpy as np
import logging
import threading
import math
from collections import deque

try:
    import torch
except ImportError:
    torch = None

from video_reader import VideoReader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_DIR_NAME = "best-512_ncnn_model"
MODEL_DIR_NAME ="weights-512-70/best-512-70_ncnn_model"
RESULT_DIR = os.path.join(BASE_DIR, "result")
TEMP_DIR = os.path.join(BASE_DIR, "temp")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
PROCESSED_DIR = os.path.join(OUTPUT_DIR, "processed")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
LOG_FILE_PATH = os.path.join(BASE_DIR, "log.txt")
EVENT_LOG_FILENAME = "events_log.json"
EVENT_LOG_AUTOCLEAR = False
EVENT_LOG_FLUSH = True
EVENT_LOG_FSYNC = True
SAVE_RESULT_REPORT = False
CPU_THREADS = int(os.getenv("ANBAR_THREADS") or 4)

if SAVE_RESULT_REPORT:
    os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="vision_id=anbar_tolid %(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("paper_roll_counter")

processing_thread = None
processing_stop_event = threading.Event()
processing_status = {
    "state": "idle",
    "started_at": None,
    "finished_at": None,
    "last_report": None,
    "last_error": None,
    "run_log": None
}

def to_rel_path(path):
    return os.path.relpath(path, BASE_DIR).replace("/", "\\")

# ====================== Performance Settings ======================
def apply_performance_settings():
    thread_val = str(CPU_THREADS)
    os.environ["OMP_NUM_THREADS"] = thread_val
    os.environ["MKL_NUM_THREADS"] = thread_val
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
    cv2.setNumThreads(CPU_THREADS)
    cv2.setUseOptimized(True)
    if torch is not None:
        try:
            torch.set_num_threads(CPU_THREADS)
        except Exception:
            pass

apply_performance_settings()

# ====================== Kalman Filter ======================
class KalmanFilter:
    """Simple 2D Constant Velocity Kalman Filter for centroid tracking"""
    def __init__(self, initial_x, initial_y):
        self.state = np.array([initial_x, initial_y, 0.0, 0.0], dtype=np.float32)
        self.F = np.array([[1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float32)
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

# ====================== FPS Calculator ======================
class FPSCalculator:
    def __init__(self, smoothing_frames=30):
        self.smoothing_frames = smoothing_frames
        self.timestamps = deque(maxlen=smoothing_frames)

    def tick(self):
        now = time.time()
        self.timestamps.append(now)
        if len(self.timestamps) >= 2:
            duration = self.timestamps[-1] - self.timestamps[0]
            if duration > 0:
                return (len(self.timestamps) - 1) / duration
        return 0.0

def save_snapshot(frame, frame_number, event_tag, target_dir=RESULT_DIR):
    timestamp_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"snapshot_{frame_number}_{event_tag}_{timestamp_tag}.jpg"
    os.makedirs(target_dir, exist_ok=True)
    snapshot_path = os.path.join(target_dir, filename)
    if cv2.imwrite(snapshot_path, frame):
        return to_rel_path(snapshot_path)
    return None

def build_counts_snapshot(primary, exits, forklift_entry, forklift_exit, paper_roll_entry, paper_roll_exit, live_count, class_name):
    return {
        "primary": primary,
        "exits": exits,
        "total": primary,
        "forklift_entry": forklift_entry,
        "forklift_exit": forklift_exit,
        "paper_roll_entry": paper_roll_entry,
        "paper_roll_exit": paper_roll_exit,
        "live": live_count,
        "last_non_forklift_primary_class": class_name #class_name last_class
    }

# ===== Configuration =====
MODEL_PATH = os.path.join(BASE_DIR, MODEL_DIR_NAME)
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, "weights", MODEL_DIR_NAME)
MODEL_DISPLAY_PATH = os.path.relpath(MODEL_PATH, BASE_DIR)

VIDEO_SOURCE = "video_source/pending/14.mp4"
OUTPUT_PATH = None
VIDEO_SOURCE_PATH = os.path.join(BASE_DIR, VIDEO_SOURCE)
OUTPUT_PATH_REL = to_rel_path(OUTPUT_PATH) if OUTPUT_PATH else ""

IMG_SIZE = 512
FRAME_SKIP = 2
CONF_THRESH = 0.4
PROCESS_MAX_WIDTH = 720
TRACKING_PERSISTENCE = 20          # max missed frames
MATCH_DISTANCE = 70                # max distance to match with Kalman prediction
SHOW_DISPLAY = True
DRAW_BOXES = True
ENABLE_OUTPUT_VIDEO = False
ENABLE_RUN_LOG = False
SAVE_EVENT_SNAPSHOTS = True
SAVE_FORKLIFT_SNAPSHOTS = True

# Colors
COLOR_BBOX_DEFAULT = (0, 255, 0)      # green - before counting
COLOR_BBOX_COUNTED = (0, 255, 255)    # yellow - after counting
COLOR_LINE = (0, 255, 255)            # yellow for counting line
COLOR_CENTER = (255, 255, 0)          # light blue for center

def _process_video():
    global processing_status
    processing_status["state"] = "running"
    processing_status["started_at"] = datetime.now().isoformat()
    processing_status["finished_at"] = None
    processing_status["last_error"] = None
    processing_status["last_report"] = None

    model = YOLO(MODEL_PATH, task="detect")
    try:
        model.fuse()
        logger.info("Model layers fused for faster inference")
    except Exception:
        logger.debug("Model fuse skipped", exc_info=True)

    reader_source = VIDEO_SOURCE if VIDEO_SOURCE == "picamera2" else VIDEO_SOURCE_PATH
    video_reader = None
    try:
        video_reader = VideoReader(reader_source)
    except ValueError as err:
        msg = str(err)
        print(f" {msg}")
        processing_status["state"] = "error"
        processing_status["last_error"] = msg
        return

    video_reader.start()
    props = video_reader.get_properties()
    width = int(props.get("width") or 0)
    height = int(props.get("height") or 0)
    fps = props.get("fps") or 30.0
    total_frames = int(props.get("total_frames") or 0)

    print(f"Video source: {reader_source}")
    print(f"Input resolution: {width}x{height}, FPS: {fps:.1f}, Total frames: {total_frames}")

    # Counting line (same as original code: center of frame)
    line_x = (2 * width) // 4 if width else 0

    # Tracker state
    tracks = {}                    # track_id -> dict
    next_track_id = 0
    entry_count = 0
    exit_count = 0
    processed_frames = 0
    forklift_entry_count = False
    forklift_exit_count = False
    paper_roll_entry_count = 0
    paper_roll_exit_count = 0
    live_count = 0
    last_non_forklift_class = None

    fps_calc = FPSCalculator()

    run_start_dt = datetime.now()
    run_stamp = run_start_dt.strftime('%Y%m%d_%H%M%S')
    jsonl_path = os.path.join(TEMP_DIR, EVENT_LOG_FILENAME)
    report_path = os.path.join(RESULT_DIR, f"paper_roll_report_{run_stamp}.json") if SAVE_RESULT_REPORT else None
    if EVENT_LOG_AUTOCLEAR:
        open(jsonl_path, "w", encoding="utf-8").close()
    jsonl_file = open(jsonl_path, "a", encoding="utf-8", buffering=1)

    report_data = {
        "device_id": "anbar_tolid",
        "video_input": reader_source,
        "model_path": MODEL_DISPLAY_PATH,
        "start_time": run_start_dt.isoformat(),
        "config": {"img_size": IMG_SIZE, "frame_skip": FRAME_SKIP, "conf_thresh": CONF_THRESH},
        "events": [],
        "statistics": {"total_entries": 0, "total_exits": 0, "objects_detected": {}}
    }

    out = None
    if ENABLE_OUTPUT_VIDEO and OUTPUT_PATH and width and height:
        out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'XVID'), fps / max(1, FRAME_SKIP), (width, height))

    try:
        while not processing_stop_event.is_set():
            frame_data = video_reader.read()
            if frame_data is None:
                if not video_reader.more():
                    break
                continue

            frame = frame_data["frame"]
            frame_count = frame_data["frame_number"]

            if (width == 0 or height == 0) and frame is not None:
                height, width = frame.shape[:2]
                line_x = (2 * width) // 4

            # Skip frames
            if FRAME_SKIP > 1 and frame_count % FRAME_SKIP != 0:
                if out:
                    out.write(frame)
                continue

            processed_frames += 1
            fps_calc.tick()
            current_fps = fps_calc.tick()

            # Preprocess for inference
            inference_frame = frame
            scale_factor = 1.0
            if PROCESS_MAX_WIDTH and width > PROCESS_MAX_WIDTH:
                scale_factor = PROCESS_MAX_WIDTH / float(width)
                inference_frame = cv2.resize(frame, (PROCESS_MAX_WIDTH, int(height * scale_factor)), interpolation=cv2.INTER_AREA)

            # Detection (without built-in tracker)
            results = model(inference_frame, imgsz=IMG_SIZE, conf=CONF_THRESH, verbose=False, max_det=8)[0]

            detections = []
            if results.boxes is not None and len(results.boxes) > 0:
                boxes = results.boxes.xyxy.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy().astype(int)
                confs = results.boxes.conf.cpu().numpy()
                for box, cls_id, conf in zip(boxes, classes, confs):
                    if scale_factor != 1.0:
                        box /= scale_factor
                    x1, y1, x2, y2 = map(int, box)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    class_name = model.names[cls_id]
                    detections.append({
                        "cx": cx, "cy": cy, "bbox": (x1, y1, x2, y2),
                        "class_name": class_name.lower(), "conf": float(conf)
                    })

            # === Kalman-based Tracking ===
            # Predict all tracks
            for track in tracks.values():
                track["kalman"].predict()

            # Match detections to tracks
            matched_track_ids = set()
            for det in detections:
                cx, cy = det["cx"], det["cy"]
                best_id = None
                best_dist = MATCH_DISTANCE
                for tid, track in tracks.items():
                    pred_x, pred_y = track["kalman"].get_predicted_position()
                    dist = math.hypot(cx - pred_x, cy - pred_y)
                    if dist < best_dist:
                        best_dist = dist
                        best_id = tid
                if best_id is not None:
                    track = tracks[best_id]
                    matched_track_ids.add(best_id)
                else:
                    # New track
                    next_track_id += 1
                    best_id = next_track_id
                    tracks[best_id] = {
                        "kalman": KalmanFilter(cx, cy),
                        "prev_cx": cx,
                        "prev_prev_cx": cx,
                        "class_name": det["class_name"],
                        "bbox": det["bbox"],          # store bbox
                        "counted": False,             # whether already counted?
                        "missed": 0,
                        "last_seen": frame_count
                    }
                    matched_track_ids.add(best_id)

                # Update track
                track = tracks[best_id]
                smooth_pos = track["kalman"].update([cx, cy])
                smooth_cx, smooth_cy = smooth_pos[0], smooth_pos[1]

                track["prev_prev_cx"] = track["prev_cx"]
                track["prev_cx"] = smooth_cx
                track["class_name"] = det["class_name"]
                track["bbox"] = det["bbox"]                # update bbox
                track["missed"] = 0
                track["last_seen"] = frame_count

                # Crossing detection (original rules preserved)
                prev_x = track.get("prev_prev_cx", smooth_cx)
                crossed_entry = prev_x < line_x <= smooth_cx
                crossed_exit = prev_x > line_x >= smooth_cx




                if crossed_entry or crossed_exit:
                    is_forklift = det["class_name"] == "forklift"
                    forklift_entry_count = True
                    forklift_exit_count = False
                    # forklift_entry_count = True
                    # forklift_exit_count = False
                    if crossed_entry:
                        entry_count += 1
                        direction = "left_to_right"
                        forklift_entry_count = True
                        forklift_exit_count = False
                        event_type = "forklift_entry" if  is_forklift  else  "paper_roll_entry"
                        if is_forklift:
                            forklift_entry_count = True
                            forklift_exit_count = False
                            live_count = 1
                        else:
                            paper_roll_entry_count += 1
                            last_non_forklift_class = det["class_name"]
                            live_count = 1
                            forklift_entry_count = True
                            forklift_exit_count = False
                    else:
                        exit_count += 1
                        direction = "right_to_left"
                        forklift_exit_count = True
                        forklift_entry_count = False
                        event_type = "forklift_exit" if is_forklift else "paper_roll_exit"
                        if is_forklift:
                            forklift_exit_count = True
                            forklift_entry_count = False
                            live_count = 1
                        else:
                            paper_roll_exit_count += 1
                            live_count = 1
                            forklift_exit_count = True
                            forklift_entry_count = False

                    # <<< Important: mark this object as counted >>>
                    track["counted"] = True
                    counts_snapshot = build_counts_snapshot(
                        entry_count, exit_count,
                        forklift_entry_count, forklift_exit_count,
                        paper_roll_entry_count, paper_roll_exit_count,
                        live_count, last_non_forklift_class
                    )

                    snapshot_rel = None
                    if SAVE_EVENT_SNAPSHOTS and (not is_forklift or SAVE_FORKLIFT_SNAPSHOTS):
                        snapshot_rel = save_snapshot(frame, frame_count, event_type, PROCESSED_DIR)

                    # Logging & reporting (same as original code)
                    jsonl_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "frame_number": frame_count,
                        "track_id": best_id,
                        "class_name": det["class_name"],
                        "line": "primary",
                        "direction": direction,
                        "event_type": event_type,
                        "snapshot_path": snapshot_rel or "",
                        "counts": counts_snapshot,
                        "device_id": "anbar_tolid"
                    }
                    jsonl_file.write(json.dumps(jsonl_entry) + "\n")
                    if EVENT_LOG_FLUSH:
                        jsonl_file.flush()
                        if EVENT_LOG_FSYNC:
                            os.fsync(jsonl_file.fileno())

                    report_data["events"].append({
                        "frame_number": frame_count,
                        "timestamp": datetime.now().isoformat(),
                        "counts": counts_snapshot,
                        "snapshot_path": snapshot_rel
                    })

                    if det["class_name"] not in report_data["statistics"]["objects_detected"]:
                        report_data["statistics"]["objects_detected"][det["class_name"]] = 0
                    report_data["statistics"]["objects_detected"][det["class_name"]] += 1
                # else:
                #     forklift_entry_count = False
                #     forklift_exit_count = True
            # Purge stale tracks
            stale = [tid for tid, t in tracks.items() if tid not in matched_track_ids]
            for tid in stale:
                tracks[tid]["missed"] += 1
                if tracks[tid]["missed"] > TRACKING_PERSISTENCE:
                    del tracks[tid]

            # Drawing line
            if line_x:
                cv2.line(frame, (line_x, 0), (line_x, height), COLOR_LINE, 3)

            # Draw all active tracks
            for tid, track in tracks.items():
                bbox = track.get("bbox")
                if bbox is None:
                    continue
                x1, y1, x2, y2 = bbox
                cx, cy = track["kalman"].get_predicted_position()

                # select color based on counted
                color = COLOR_BBOX_COUNTED if track.get("counted", False) else COLOR_BBOX_DEFAULT

                # draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # draw center and label
                cv2.circle(frame, (cx, cy), 8, COLOR_CENTER, -1)
                label = f"{track['class_name'].capitalize()} ID:{tid}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Overlay info
            cv2.rectangle(frame, (5, 5), (300, 160), (40, 40, 40), -1)
            cv2.putText(frame, f"Entries: {entry_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 50), 2)
            cv2.putText(frame, f"Exits: {exit_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 255), 2)
            cv2.putText(frame, f"Live: {live_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 215, 0), 2)
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 255), 2)

            if out:
                out.write(frame)
            if SHOW_DISPLAY:
                display_frame = cv2.resize(frame, (960, 540))
                cv2.imshow("paper_roll Counter", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    processing_stop_event.set()
                    break
                elif key == ord('p'):
                    while cv2.waitKey(1) & 0xFF not in [ord('p'), ord('q')]:
                        pass
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        processing_stop_event.set()
                        break

            if frame_count % 10 == 0:
                progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
                print(f"Processing: {frame_count}/{total_frames} ({progress:.1f}%) | FPS: {current_fps:.1f}")

    except Exception as e:
        processing_status["state"] = "error"
        processing_status["last_error"] = str(e)
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup and reporting (same as original code)
        if video_reader:
            video_reader.stop()
        if out:
            out.release()
        if SHOW_DISPLAY:
            cv2.destroyAllWindows()
        if jsonl_file:
            jsonl_file.close()

        total_time = time.time() - run_start_dt.timestamp()
        counts_summary = build_counts_snapshot(
            entry_count, exit_count,
            forklift_entry_count, forklift_exit_count,
            paper_roll_entry_count, paper_roll_exit_count,
            live_count, last_non_forklift_class
        )
        report_data["statistics"]["total_entries"] = entry_count
        report_data["statistics"]["total_exits"] = exit_count
        report_data["end_time"] = datetime.now().isoformat()
        report_data["processing_time_seconds"] = total_time
        report_data["average_fps"] = processed_frames / total_time if total_time > 0 else 0
        report_data["total_frames"] = total_frames if total_frames > 0 else frame_count
        report_data["frames_processed"] = processed_frames
        report_data["counts"] = counts_summary

        if SAVE_RESULT_REPORT and report_path:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2)

        print("\n" + "="*50)
        print(f"Video processing completed in {total_time:.1f} seconds")
        print(f"Entries: {entry_count} | Exits: {exit_count} | Live: {live_count}")
        if SAVE_RESULT_REPORT and report_path:
            print(f"Report saved to: {to_rel_path(report_path)}")
        print("="*50)

        processing_status["state"] = "stopped" if processing_stop_event.is_set() else "completed"
        processing_status["last_report"] = to_rel_path(report_path) if (SAVE_RESULT_REPORT and report_path) else None
        processing_status["finished_at"] = datetime.now().isoformat()

# same start/stop/status functions
def start():
    global processing_thread, processing_stop_event
    if processing_thread and processing_thread.is_alive():
        return False
    processing_stop_event = threading.Event()
    processing_thread = threading.Thread(target=_process_video, daemon=True)
    processing_thread.start()
    return True

def stop(wait=False):
    global processing_thread
    if not (processing_thread and processing_thread.is_alive()):
        return False
    processing_stop_event.set()
    if wait:
        processing_thread.join()
    return True

def status():
    info = processing_status.copy()
    info["running"] = processing_thread.is_alive() if processing_thread else False
    return info

if __name__ == "__main__":
    if start():
        processing_thread.join()