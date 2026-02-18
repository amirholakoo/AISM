from ultralytics import YOLO
import cv2
import json
from datetime import datetime
import os
import time
import numpy as np
import logging
import threading
try:
    import torch
except ImportError:
    torch = None
from video_reader import VideoReader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#MODEL_DIR_NAME = "best-50-new-yolov11_ncnn_model"

MODEL_DIR_NAME = "best_70_new_ncnn_model_with_auggmentation"

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
CPU_THREADS = int(os.getenv("ANBAR_THREADS") or 4)

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


def configure_runtime():
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


configure_runtime()


def save_snapshot(frame, frame_number, event_tag, target_dir=RESULT_DIR):
    timestamp_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"snapshot_{frame_number}_{event_tag}_{timestamp_tag}.jpg"
    os.makedirs(target_dir, exist_ok=True)
    snapshot_path = os.path.join(target_dir, filename)
    if cv2.imwrite(snapshot_path, frame):
        return to_rel_path(snapshot_path)
    return None


def build_counts_snapshot(primary, exits, forklift_entry, forklift_exit, paper_roll_entry, paper_roll_exit, live_count, last_class):
    return {
        "primary": primary,
        "exits": exits,
        "total": primary,
        "forklift_entry": forklift_entry,
        "forklift_exit": forklift_exit,
        "paper_roll_entry": paper_roll_entry,
        "paper_roll_exit": paper_roll_exit,
        "live": live_count,
        "last_non_forklift_primary_class": last_class
    }

# ===== Optimization settings =====
# MODEL_PATH = "../forklift_counting/weights/best.pt"

# MODEL_PATH = "../weights/video_source/best-50-new-yolov11_ncnn_model.pt"
MODEL_PATH = os.path.join(BASE_DIR, MODEL_DIR_NAME)
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, "weights", MODEL_DIR_NAME)
MODEL_DISPLAY_PATH = os.path.relpath(MODEL_PATH, BASE_DIR)

# VIDEO_SOURCE = "../forklift_counting/video_source/pending/39.mp4"
VIDEO_SOURCE = "video_source/pending/10.mp4"
# OUTPUT_PATH = os.path.join(PROCESSED_DIR, "output.mp4")
OUTPUT_PATH = None
VIDEO_SOURCE_PATH = os.path.join(BASE_DIR, VIDEO_SOURCE)
OUTPUT_PATH_REL = to_rel_path(OUTPUT_PATH) if OUTPUT_PATH else ""

# Critical parameters for performance and quality
IMG_SIZE = 640           # Balanced resolution to preserve quality
FRAME_SKIP = 2         # Processes every frame
CONF_THRESH = 0.6        # Balanced confidence threshold
PROCESS_MAX_WIDTH = 720  # Set to 0 to disable downscaling before inference
TRACKING_PERSISTENCE = 20
SHOW_DISPLAY = True
DRAW_BOXES = True
ENABLE_OUTPUT_VIDEO = False
ENABLE_RUN_LOG = False
SAVE_EVENT_SNAPSHOTS = True
SAVE_FORKLIFT_SNAPSHOTS = True
REPORT_EVENTS_DESCENDING = True

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
    fps = props.get("fps") or 0.0
    total_frames = int(props.get("total_frames") or 0)
    writer_fps = fps if fps and fps > 0 else 30.0
    print(f"Video source: {reader_source}")
    print(f"Input resolution: {width or 'unknown'}x{height or 'unknown'}, FPS: {(fps or 0):.1f}, Total frames: {total_frames or 'unknown'}")
    logger.info(
        "paper_roll CountingApp initialized input=%s model=%s width=%s height=%s fps=%s",
        reader_source,
        MODEL_DISPLAY_PATH,
        width or "unknown",
        height or "unknown",
        fps or "unknown"
    )

    out = None
    line_x = (2 * width) // 4 if width else None
    object_tracker = {}
    entry_count = 0
    exit_count = 0
    frame_count = 0
    processed_frames = 0
    forklift_entry_count = 0
    forklift_exit_count = 0
    paper_roll_entry_count = 0
    paper_roll_exit_count = 0
    live_count = 0
    last_non_forklift_class = None
    jsonl_file = None
    run_log_file = None
    jsonl_rel_path = ""
    report_rel_path = ""
    run_log_rel_path = ""

    run_start_dt = datetime.now()
    run_stamp = run_start_dt.strftime('%Y%m%d_%H%M%S')
    report_basename = f"paper_roll_report_{run_stamp}"
    jsonl_path = os.path.join(TEMP_DIR, EVENT_LOG_FILENAME)
    report_path = os.path.join(RESULT_DIR, f"{report_basename}.json")
    run_log_path = os.path.join(BASE_DIR, f"log_{run_stamp}.txt")
    if EVENT_LOG_AUTOCLEAR:
        open(jsonl_path, "w", encoding="utf-8").close()
    jsonl_file = open(jsonl_path, "a", encoding="utf-8", buffering=1)
    run_log_file = open(run_log_path, "w", encoding="utf-8") if ENABLE_RUN_LOG else None
    jsonl_rel_path = to_rel_path(jsonl_path)
    run_log_rel_path = to_rel_path(run_log_path) if ENABLE_RUN_LOG else ""
    processing_status["run_log"] = run_log_rel_path if ENABLE_RUN_LOG else None
    logger.info("Run started")

    def write_run_log(message):
        if not (ENABLE_RUN_LOG and run_log_file):
            return
        timestamp = datetime.now().isoformat()
        run_log_file.write(f"{timestamp} device_id=anbar_tolid {message}\n")
        run_log_file.flush()

    report_data = {
        "device_id": "anbar_tolid",
        "video_input": reader_source,
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
    print(" Starting video processing...")
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
                line_x = (3 * width) // 4 if width else 0
            frame_h, frame_w = frame.shape[:2]
            inference_frame = frame
            scale_factor = 1.0
            if PROCESS_MAX_WIDTH and frame_w > PROCESS_MAX_WIDTH:
                scale_factor = PROCESS_MAX_WIDTH / float(frame_w)
                resized_height = max(1, int(frame_h * scale_factor))
                inference_frame = cv2.resize(
                    frame,
                    (PROCESS_MAX_WIDTH, resized_height),
                    interpolation=cv2.INTER_AREA
                )
            if ENABLE_OUTPUT_VIDEO and out is None and width and height and OUTPUT_PATH:
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
                source=inference_frame,
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
                live_count = 0
                for i in range(len(boxes)):
                    box = boxes[i].copy()
                    if scale_factor != 1.0:
                        box /= scale_factor
                    obj_id = ids[i]
                    cls_id = classes[i]
                    conf = confs[i]
                    
                    x1, y1, x2, y2 = map(int, box)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    class_name = model.names[int(cls_id)]
                    is_forklift = class_name.lower() == "forklift"
                    
                    if DRAW_BOXES:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                        label = f"{class_name} {conf:.2f}"
                        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 200, 0), -1)
                        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    prev_data = object_tracker.get(obj_id)
                    prev_x = prev_data["position"] if prev_data else None
                    crossed_entry = prev_x is not None and prev_x < line_x and cx >= line_x
                    crossed_exit = prev_x is not None and prev_x > line_x and cx <= line_x

                    if crossed_entry or crossed_exit:
                        event_timestamp = datetime.now()
                        if crossed_entry:
                            entry_count += 1
                            line_label = "primary"
                            direction = "left_to_right"
                            if is_forklift:
                                forklift_entry_count = True
                                forklift_exit_count = False
                                event_type = "forklift_entry"
                                live_count = 1
                            else:
                                forklift_entry_count = True
                                forklift_exit_count = False
                                paper_roll_entry_count += 1
                                event_type = "paper_roll_entry"
                                last_non_forklift_class = class_name
                                print("live_count", live_count)
                                live_count += 1
                        else:
                            exit_count += 1
                            line_label = "primary"
                            direction = "right_to_left"
                            if is_forklift:
                                forklift_exit_count = True
                                forklift_entry_count = False
                                event_type = "forklift_exit"
                                live_count = 1
                            else:
                                forklift_exit_count = True
                                forklift_entry_count = False
                                paper_roll_exit_count += 1
                                event_type = "paper_roll_exit"
                                live_count += 1
                        counts_snapshot = build_counts_snapshot(
                            entry_count,
                            exit_count,
                            forklift_entry_count,
                            forklift_exit_count,
                            paper_roll_entry_count,
                            paper_roll_exit_count,
                            live_count,
                            last_non_forklift_class
                        )
                        snapshot_rel = None
                        if SAVE_EVENT_SNAPSHOTS and (not is_forklift or SAVE_FORKLIFT_SNAPSHOTS):
                            snapshot_rel = save_snapshot(frame, frame_count, event_type, PROCESSED_DIR)
                            if snapshot_rel:
                                logger.info(
                                    "Snapshot saved type=%s frame=%d path=%s",
                                    event_type,
                                    frame_count,
                                    snapshot_rel
                                )
                        event_record = {
                            "frame_number": frame_count,
                            "timestamp": event_timestamp.isoformat(),
                            "counts": counts_snapshot,
                            "snapshot_path": snapshot_rel
                        }
                        if REPORT_EVENTS_DESCENDING:
                            report_data["events"].insert(0, event_record)
                        else:
                            report_data["events"].append(event_record)
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
                            "device_id": "anbar_tolid"
                        }
                        if jsonl_file:
                            jsonl_file.write(json.dumps(jsonl_entry) + "\n")
                            if EVENT_LOG_FLUSH:
                                jsonl_file.flush()
                                if EVENT_LOG_FSYNC:
                                    os.fsync(jsonl_file.fileno())
                        event_log_line = (
                            f"type={event_type} frame={frame_count} "
                            f"entries={entry_count} exits={exit_count}"
                        )
                        write_run_log(event_log_line)
                        logger.info(
                            "Real-time event JSON logged type=%s frame=%d path=%s",
                            event_type,
                            frame_count,
                            jsonl_rel_path
                        )
                    
                    object_tracker[obj_id] = {
                        "position": cx,
                        "center": (cx, cy),
                        "class": class_name,
                        "last_seen": frame_count
                    }
                    
                    if class_name not in report_data["statistics"]["objects_detected"]:
                        report_data["statistics"]["objects_detected"][class_name] = 0
                    report_data["statistics"]["objects_detected"][class_name] += 1
            
            current_ids = list(object_tracker.keys())
            for obj_id in current_ids:
                if frame_count - object_tracker[obj_id].get("last_seen", 0) > TRACKING_PERSISTENCE:
                    del object_tracker[obj_id]
            
            if line_x is not None:
                cv2.line(frame, (line_x, 0), (line_x, height), (0, 255, 255), 3)
            
            if SHOW_DISPLAY:
                cv2.rectangle(frame, (5, 5), (300, 160), (40, 40, 40), -1)
                cv2.putText(frame, f"Entries: {entry_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 50), 2)
                cv2.putText(frame, f"Exits: {exit_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 255), 2)
                cv2.putText(frame, f"Live: {live_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 215, 0), 2)
            
            if current_time - prev_time >= 1.0:
                avg_fps = fps_counter / (current_time - prev_time)
                prev_time = current_time
                fps_counter = 0
                
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 255), 2)
            # processing_time = time.time() - frame_start
            # cv2.putText(frame, f"Proc: {processing_time*1000:.1f}ms", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 100), 2)
            # if total_frames > 0:
            #     progress = (frame_count / total_frames) * 100
            #     cv2.putText(frame, f"Progress: {progress:.1f}%", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
            
            if ENABLE_OUTPUT_VIDEO and out is not None:
                out.write(frame)            
            if SHOW_DISPLAY:
                display_frame = cv2.resize(frame, (960, 540))
                cv2.imshow("paper_roll Counter", display_frame)
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
                    print(f"Processing: {frame_count}/{total_frames} frames ({progress:.1f}%) | FPS: {avg_fps:.1f}")
                else:
                    print(f"Processing: {frame_count} frames (progress unknown) | FPS: {avg_fps:.1f}")

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
        if jsonl_file:
            jsonl_file.close()

        total_time = time.time() - loop_start
        counts_summary = build_counts_snapshot(
            entry_count,
            exit_count,
            forklift_entry_count,
            forklift_exit_count,
            paper_roll_entry_count,
            paper_roll_exit_count,
            live_count,
            last_non_forklift_class
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
        if ENABLE_OUTPUT_VIDEO and OUTPUT_PATH_REL:
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
    if start():
        processing_thread.join()
