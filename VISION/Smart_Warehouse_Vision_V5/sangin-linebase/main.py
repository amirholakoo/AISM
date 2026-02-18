from ultralytics import YOLO
import cv2
import json
from datetime import datetime
import os
import time
import numpy as np
import logging
import threading
from config import Config
from video_reader import VideoReader

config = Config()
BASE_DIR = config.BASE_DIR
RESULT_DIR = config.RESULT_DIR
TEMP_DIR = config.TEMP_DIR
OUTPUT_DIR = config.OUTPUT_DIR
PROCESSED_DIR = config.PROCESSED_DIR
LOG_DIR = config.LOG_DIR
LOG_FILE_PATH = config.LOG_FILE_PATH

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="vision_id=anbar_sangin %(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("forklift_counter")
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


def save_snapshot(frame, frame_number, count_value, event_timestamp=None, class_name=None, event_type=None):
    """
    Save a snapshot with timestamp in filename and overlay text on the image.

    Overlay includes:
      - event timestamp
      - class name
      - ENTRY/EXIT
    """
    ts = event_timestamp or datetime.now()
    timestamp_tag = ts.strftime("%Y%m%d_%H%M%S")
    event_label = ""
    if isinstance(event_type, str):
        if event_type.lower() == "entry":
            event_label = "ENTRY"
        elif event_type.lower() == "exit":
            event_label = "EXIT"
        else:
            event_label = event_type.upper()

    safe_class = (class_name or "unknown").replace(" ", "_")
    filename = f"snapshot_{frame_number}_{count_value}_{safe_class}_{event_label}_{timestamp_tag}.jpg"
    snapshot_path = os.path.join(RESULT_DIR, filename)

    # Draw overlay text on a copy so we don't mutate the live frame
    img = frame.copy()
    overlay_1 = f"{ts.strftime('%Y-%m-%d %H:%M:%S')} | {event_label}".strip(" |")
    overlay_2 = f"class: {class_name}" if class_name else ""
    overlay_3 = f"count: {count_value}"

    x, y = 10, 28
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2

    def draw_text(line, yy):
        if not line:
            return
        # black outline for readability
        cv2.putText(img, line, (x, yy), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(img, line, (x, yy), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    draw_text(overlay_1, y)
    draw_text(overlay_2, y + 26)
    draw_text(overlay_3, y + 52)

    if cv2.imwrite(snapshot_path, img):
        return to_rel_path(snapshot_path)
    return None


def build_counts_snapshot(primary, exits, forklift_entry, forklift_exit, live_status):
    return {
        "primary": primary,
        "exits": exits,
        "total": primary,
        "forklift_entry": forklift_entry,
        "forklift_exit": forklift_exit,
        "live": live_status,
    }


def resize_frame_for_display(frame, max_width, max_height):
    """Resize frame preserving aspect ratio so it fits within max bounds."""
    if frame is None:
        return frame
    if max_width is None or max_height is None:
        return frame
    if max_width <= 0 or max_height <= 0:
        return frame
    height, width = frame.shape[:2]
    if width == 0 or height == 0:
        return frame
    scale = min(max_width / width, max_height / height)
    if abs(scale - 1.0) < 1e-3:
        return frame
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    return cv2.resize(frame, (new_width, new_height))

# ===== Optimization Settings =====
MODEL_PATH = config.MODEL_PATH
MODEL_DISPLAY_PATH = os.path.relpath(MODEL_PATH, BASE_DIR)

VIDEO_SOURCE = config.VIDEO_SOURCE
VIDEO_SOURCE_PATH = (
    VIDEO_SOURCE if VIDEO_SOURCE == "picamera2" else os.path.join(BASE_DIR, VIDEO_SOURCE)
)
# Video output is disabled
ENABLE_OUTPUT_VIDEO = False

if ENABLE_OUTPUT_VIDEO and config.VIDEO_OUTPUT_NAME:
    OUTPUT_PATH = os.path.join(PROCESSED_DIR, config.VIDEO_OUTPUT_NAME)
    OUTPUT_PATH_REL = to_rel_path(OUTPUT_PATH)
else:
    OUTPUT_PATH = ""
    OUTPUT_PATH_REL = ""

# Critical runtime parameters
IMG_SIZE = config.DETECTION_IMAGE_SIZE           # Balanced resolution to preserve detail
FRAME_SKIP = config.FRAME_SKIP                   # Process one out of every N frames
CONF_THRESH = config.CONFIDENCE_THRESHOLD        # Detection confidence threshold
TRACKING_PERSISTENCE = config.TRACKING_PERSISTENCE
COUNTING_ZONE_MARGIN = config.COUNTING_ZONE_MARGIN  # Margin around counting line
SHOW_DISPLAY = config.SHOW_DISPLAY
DRAW_BOXES = config.DRAW_BOXES
ENABLE_RUN_LOG = config.ENABLE_RUN_LOG
DISPLAY_WINDOW_NAME = config.DISPLAY_WINDOW_NAME
DISPLAY_MAX_WIDTH = config.DISPLAY_MAX_WIDTH
DISPLAY_MAX_HEIGHT = config.DISPLAY_MAX_HEIGHT
PICAMERA_OPTIONS = {
    "main_size": config.PICAMERA_MAIN_SIZE,
    "main_format": config.PICAMERA_MAIN_FORMAT,
    "startup_delay": config.PICAMERA_STARTUP_DELAY,
}
if getattr(config, "PICAMERA_SENSOR_CONFIG", None):
    PICAMERA_OPTIONS["sensor_config"] = config.PICAMERA_SENSOR_CONFIG

# Advanced CPU settings
os.environ["OMP_NUM_THREADS"] = config.OMP_NUM_THREADS
os.environ["MKL_NUM_THREADS"] = config.MKL_NUM_THREADS
os.environ["OMP_DYNAMIC"] = config.OMP_DYNAMIC
os.environ["KMP_AFFINITY"] = config.KMP_AFFINITY
cv2.setNumThreads(1)  # Optimal threading for OpenCV

def _process_video():
    global processing_status
    processing_status["state"] = "running"
    processing_status["started_at"] = datetime.now().isoformat()
    processing_status["finished_at"] = None
    processing_status["last_error"] = None
    processing_status["last_report"] = None

    print("Loading PyTorch model...")
    model = YOLO(MODEL_PATH)
    backend_model = getattr(model, "model", None)
    if hasattr(backend_model, "parameters"):
        for param in backend_model.parameters():
            param.requires_grad = False

    reader_source = VIDEO_SOURCE if VIDEO_SOURCE == "picamera2" else VIDEO_SOURCE_PATH
    actual_reader_source = reader_source
    video_reader = None
    try:
        video_reader = VideoReader(
            reader_source,
            queue_size=config.QUEUE_SIZE,
            picamera_options=dict(PICAMERA_OPTIONS)
        )
    except ValueError as err:
        msg = str(err)
        print(f"Error: {msg}")
        processing_status["state"] = "error"
        processing_status["last_error"] = msg
        return

    video_reader.start()
    props = video_reader.get_properties()
    if video_reader.picam2 is not None:
        actual_reader_source = "picamera2"
    width = int(props.get("width") or 0)
    height = int(props.get("height") or 0)
    fps = props.get("fps") or 0.0
    total_frames = int(props.get("total_frames") or 0)
    writer_fps = fps if fps and fps > 0 else 30.0
    print(f"Video source: {actual_reader_source}")
    print(
        f"Input resolution: {width or 'unknown'}x{height or 'unknown'}, "
        f"FPS: {(fps or 0):.1f}, Total frames: {total_frames or 'unknown'}"
    )
    logger.info(
        "Forklift CountingApp initialized input=%s model=%s width=%s height=%s fps=%s",
        actual_reader_source,
        MODEL_DISPLAY_PATH,
        width or "unknown",
        height or "unknown",
        fps or "unknown"
    )

    out = None
    line_x = width // 3 if width else None
    object_tracker = {}
    entry_count = 0
    exit_count = 0
    frame_count = 0
    processed_frames = 0
    forklift_entry_count = False
    forklift_exit_count = False
    jsonl_file = None
    run_log_file = None
    jsonl_rel_path = ""
    report_rel_path = ""
    run_log_rel_path = ""
    live_status = 1

    run_start_dt = datetime.now()
    run_stamp = run_start_dt.strftime('%Y%m%d_%H%M%S')
    report_basename = f"forklift_report_{run_stamp}"
    jsonl_path = os.path.join(TEMP_DIR, "events_log.json")
    report_path = os.path.join(RESULT_DIR, f"{report_basename}.json")
    run_log_path = os.path.join(BASE_DIR, f"log_{run_stamp}.txt")
    jsonl_file = open(jsonl_path, "w", encoding="utf-8")
    run_log_file = open(run_log_path, "w", encoding="utf-8") if ENABLE_RUN_LOG else None
    jsonl_rel_path = to_rel_path(jsonl_path)
    run_log_rel_path = to_rel_path(run_log_path) if ENABLE_RUN_LOG else ""
    processing_status["run_log"] = run_log_rel_path if ENABLE_RUN_LOG else None
    events_log_file = open(jsonl_path, "a", encoding="utf-8")
    logger.info("Run started")

    def write_run_log(message):
        if not (ENABLE_RUN_LOG and run_log_file):
            return
        timestamp = datetime.now().isoformat()
        run_log_file.write(f"{timestamp} device_id=anbar_sangin {message}\n")
        run_log_file.flush()

    report_data = {
        "device_id": "anbar_sangin",
        "video_input": actual_reader_source,
        "model_path": MODEL_DISPLAY_PATH,
        "start_time": run_start_dt.isoformat(),
        "config": {
            "img_size": IMG_SIZE,
            "frame_skip": FRAME_SKIP,
            "conf_thresh": CONF_THRESH
        },
        "events": [],
        "statistics": {
            "total_entries": 0,
            "total_exits": 0,
            "objects_detected": {}
        }
    }

    loop_start = time.time()
    prev_time = loop_start
    fps_counter = 0
    avg_fps = 0

    write_run_log("Run initialized")
    print("Starting video processing...")
    try:
        while not processing_stop_event.is_set():
            frame_data = video_reader.read()
            if frame_data is None:
                if not video_reader.more():
                    break
                continue
            
            frame = frame_data["frame"]
            frame_count = frame_data["frame_number"]
            frame_start = time.time()

            if (width == 0 or height == 0) and frame is not None:
                height, width = frame.shape[:2]
                line_x = width // 3 if width else 0
            if ENABLE_OUTPUT_VIDEO and OUTPUT_PATH and out is None and width and height:
                out = cv2.VideoWriter(
                    OUTPUT_PATH,
                    cv2.VideoWriter_fourcc(*'XVID'),
                    writer_fps / max(1, FRAME_SKIP),
                    (width, height)
                )
            
            if FRAME_SKIP > 1 and frame_count % FRAME_SKIP != 0:
                if ENABLE_OUTPUT_VIDEO and out is not None:
                    out.write(frame)
                    continue

            processed_frames += 1
            current_time = time.time()
            fps_counter += 1
            
            results = model.track(
                source=frame,
                persist=True,
                verbose=False,
                imgsz=IMG_SIZE,
                conf=CONF_THRESH,
                device='cpu',
                tracker="bytetrack.yaml",
                classes=None,
                max_det=8,
                agnostic_nms=True
            )
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                classes = results[0].boxes.cls.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                
                for i in range(len(boxes)):
                    box = boxes[i]
                    obj_id = ids[i]
                    cls_id = classes[i]
                    conf = confs[i]
                    
                    x1, y1, x2, y2 = map(int, box)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    class_name = model.names[int(cls_id)]
                    is_forklift = class_name.lower() == "forklift"
                    
                    prev_data = object_tracker.get(obj_id)
                    prev_x = prev_data["position"] if prev_data else None
                    was_counted_before = prev_data.get("counted", False) if prev_data else False
                    
                    # Calculate margin boundaries
                    margin_left = line_x - COUNTING_ZONE_MARGIN
                    margin_right = line_x + COUNTING_ZONE_MARGIN
                    
                    # Check if object exited the margin zone (for counting)
                    # Only count when object was INSIDE the margin zone and then EXITED it
                    # Entry (left to right): object was inside margin zone and exited to the right
                    was_inside_margin = prev_x is not None and margin_left <= prev_x <= margin_right
                    crossed_entry = was_inside_margin and cx > margin_right
                    # Exit (right to left): object was inside margin zone and exited to the left
                    crossed_exit = was_inside_margin and cx < margin_left

                    # Check if this object just got counted
                    just_counted = False
                    # Check for line crossing (all objects)
                    if crossed_entry or crossed_exit:
                        event_timestamp = datetime.now()
                        if crossed_entry:
                            entry_count += 1
                            line_label = "primary"
                            direction = "left_to_right" # exit
                            forklift_entry_count = True
                            forklift_exit_count = False
                            if is_forklift:
                                forklift_entry_count = True
                                forklift_exit_count = False
                                live_status = 1
                            event_type = "entry"
                        else:
                            exit_count += 1
                            line_label = "primary"
                            direction = "right_to_left" # entry
                            forklift_entry_count = False
                            forklift_exit_count = True
                            if is_forklift:
                                forklift_entry_count = False
                                forklift_exit_count = True
                                live_status = 1
                            event_type = "exit"
                        
                        counts_snapshot = build_counts_snapshot(
                            entry_count,
                            exit_count,
                            forklift_entry_count,
                            forklift_exit_count,
                            live_status,
                        )
                        snapshot_count_value = entry_count if event_type == "entry" else exit_count
                        snapshot_rel = save_snapshot(
                            frame,
                            frame_count,
                            snapshot_count_value,
                            event_timestamp=event_timestamp,
                            class_name=class_name,
                            event_type=event_type,
                        )
                        if snapshot_rel:
                            logger.info(
                                "Count increased frame=%d class=%s primary=%d exits=%d total=%d snapshot=%s",
                                frame_count,
                                class_name,
                                entry_count,
                                exit_count,
                                entry_count,
                                snapshot_rel
                            )
                        report_data["events"].append({
                            "frame_number": frame_count,
                            "timestamp": event_timestamp.isoformat(),
                            "class_name": class_name,
                            "counts": counts_snapshot,
                            "snapshot_path": snapshot_rel
                        })
                        jsonl_entry = {
                            "timestamp": event_timestamp.isoformat(),
                            "frame_number": frame_count,
                            "track_id": int(obj_id),
                            "class_name": class_name,
                            "line": line_label,
                            "direction": direction,
                            "event_type": event_type,
                            "snapshot_path": snapshot_rel,
                            "counts": counts_snapshot,
                            "device_id": "anbar_sangin"
                        }
                        events_log_file.write(json.dumps(jsonl_entry) + "\n")
                        events_log_file.flush()
                        event_log_line = (
                            f"type={event_type} class={class_name} frame={frame_count} "
                            f"entries={entry_count} exits={exit_count}"
                        )
                        print("==================================before write_run_log")
                        write_run_log(event_log_line)
                        print("==================================after write_run_log")
                        logger.info(
                            "Real-time event JSON logged type=%s class=%s frame=%d path=%s",
                            event_type,
                            class_name,
                            frame_count,
                            jsonl_rel_path
                        )
                        # Mark this object as counted
                        just_counted = True
                    
                    # Determine if object is counted (was counted before or just got counted)
                    is_counted = was_counted_before or just_counted
                    
                    # Choose color based on whether object has been counted
                    # Green (0, 200, 0) for uncounted objects, Yellow (0, 255, 255) for counted objects
                    box_color = (0, 255, 255) if is_counted else (0, 200, 0)
                    
                    if DRAW_BOXES:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                        label = f"{class_name} {conf:.2f}"
                        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), box_color, -1)
                        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Update tracker for all objects
                    object_tracker[obj_id] = {
                        "position": cx,
                        "center": (cx, cy),
                        "class": class_name,
                        "last_seen": frame_count,
                        "counted": is_counted
                    }
                    
                    # Track detected classes
                    if class_name not in report_data["statistics"]["objects_detected"]:
                        report_data["statistics"]["objects_detected"][class_name] = 0
                    report_data["statistics"]["objects_detected"][class_name] += 1
            
            current_ids = list(object_tracker.keys())
            for obj_id in current_ids:
                if frame_count - object_tracker[obj_id].get("last_seen", 0) > TRACKING_PERSISTENCE:
                    del object_tracker[obj_id]
            
            # Draw counting line and margin zone
            if line_x is not None:
                # Draw main counting line
                cv2.line(frame, (line_x, 0), (line_x, height), (0, 255, 255), 3)
                # Draw margin boundaries
                margin_left = line_x - COUNTING_ZONE_MARGIN
                margin_right = line_x + COUNTING_ZONE_MARGIN
                cv2.line(frame, (margin_left, 0), (margin_left, height), (0, 200, 200), 2)
                cv2.line(frame, (margin_right, 0), (margin_right, height), (0, 200, 200), 2)
            
            if SHOW_DISPLAY:
                cv2.rectangle(frame, (5, 5), (300, 160), (40, 40, 40), -1)
                cv2.putText(frame, f"Entries: {entry_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 50), 2)
                cv2.putText(frame, f"Exits: {exit_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 255), 2)
            
            if current_time - prev_time >= 1.0:
                avg_fps = fps_counter / (current_time - prev_time)
                prev_time = current_time
                fps_counter = 0
                
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 255), 2)
            cv2.putText(frame, f"Live: {live_status}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)
            # processing_time = time.time() - frame_start
            # cv2.putText(frame, f"Proc: {processing_time*1000:.1f}ms", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 100), 2)
            # if total_frames > 0:
            #     progress = (frame_count / total_frames) * 100
            #     cv2.putText(frame, f"Progress: {progress:.1f}%", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
            
            if ENABLE_OUTPUT_VIDEO and out is not None:
                out.write(frame)
            
            if SHOW_DISPLAY:
                display_frame = resize_frame_for_display(frame, DISPLAY_MAX_WIDTH, DISPLAY_MAX_HEIGHT)
                cv2.imshow(DISPLAY_WINDOW_NAME, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    processing_stop_event.set()
                    break
                elif key == ord('p'):
                    while True:
                        key2 = cv2.waitKey(1)
                        if key2 == ord('p') or key2 == ord('q'):
                            break
                    if key2 == ord('q'):
                        processing_stop_event.set()
                        break

            if frame_count % 10 == 0:
                if total_frames > 0:
                    progress = (frame_count / total_frames) * 100
                    print(
                        f"Processing: {frame_count}/{total_frames} frames "
                        f"({progress:.1f}%) | FPS: {avg_fps:.1f}"
                    )
                else:
                    print(
                        f"Processing: {frame_count} frames (progress unknown) | "
                        f"FPS: {avg_fps:.1f}"
                    )

    except Exception as e:
        processing_status["state"] = "error"
        processing_status["last_error"] = str(e)
        print(f"Critical error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if video_reader:
            video_reader.stop()
        if ENABLE_OUTPUT_VIDEO and out:
            out.release()
        if SHOW_DISPLAY:
            cv2.destroyAllWindows()
        if events_log_file:
            events_log_file.close()

        total_time = time.time() - loop_start
        counts_summary = build_counts_snapshot(
            entry_count,
            exit_count,
            forklift_entry_count,
            forklift_exit_count,
            live_status,
        )
        report_data["statistics"]["total_entries"] = entry_count
        report_data["statistics"]["total_exits"] = exit_count
        report_data["end_time"] = datetime.now().isoformat()
        report_data["processing_time_seconds"] = total_time
        report_data["total_processing_time"] = total_time
        report_data["average_fps"] = processed_frames / total_time if total_time > 0 else 0
        actual_total_frames = total_frames if total_frames > 0 else frame_count
        report_data["total_frames"] = actual_total_frames
        report_data["frames_processed"] = frame_count
        report_data["processed_frames"] = processed_frames
        report_data["effective_frames"] = processed_frames
        report_data["frame_skip"] = FRAME_SKIP
        report_data["counts"] = counts_summary
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2)
        report_rel_path = to_rel_path(report_path)
        logger.info("Report saved to %s", report_rel_path)
        logger.info("JSONL events saved to %s", jsonl_rel_path)
        
        os.makedirs(LOG_DIR, exist_ok=True)
        log_file = os.path.join(LOG_DIR, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2)
        log_file_rel = to_rel_path(log_file)
        
        logger.info(
            "Run finished elapsed=%.1fs frames=%d primary=%d exits=%d total=%d",
            total_time,
            frame_count,
            entry_count,
            exit_count,
            entry_count
        )
        
        print("\n" + "="*50)
        print(f"Video processing completed in {total_time:.1f} seconds")
        print(f"Total frames: {report_data['total_frames']}")
        print(f"Processed frames: {processed_frames} (Frame skip: {FRAME_SKIP})")
        print(f"Average FPS: {report_data['average_fps']:.1f}")
        print(f"Entries: {entry_count} | Exits: {exit_count}")
        if ENABLE_OUTPUT_VIDEO and OUTPUT_PATH:
            print(f"Output video: {OUTPUT_PATH_REL}")
        print(f"Report saved to: {report_rel_path}")
        print(f"Events log: {jsonl_rel_path}")
        if ENABLE_RUN_LOG:
            print(f"Run log: {run_log_rel_path}")
        print(f"Log saved to: {log_file_rel}")
        print("="*50)

        if run_log_file:
            write_run_log(
                f"Run finished elapsed={total_time:.1f}s entries={entry_count} exits={exit_count}"
            )
            write_run_log(f"Report: {report_rel_path}")
            write_run_log(f"Events log: {jsonl_rel_path}")
            write_run_log(f"Metrics log: {log_file_rel}")
            run_log_file.close()

        if processing_status["state"] != "error":
            processing_status["state"] = "stopped" if processing_stop_event.is_set() else "completed"
            processing_status["last_report"] = report_rel_path
            processing_status["finished_at"] = datetime.now().isoformat()


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
    try:
        if start():
            processing_thread.join()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received, stopping...")
        stop(wait=True)
