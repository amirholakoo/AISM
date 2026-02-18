import os
import json
import cv2
import time
import numpy as np
from pathlib import Path
from collections import deque
from datetime import datetime

# ====================== CONFIG ======================
class Config:
    """Configuration optimized for Raspberry Pi"""
    # If you have an NCNN model, convert it to PyTorch or use a small YOLO model
    MODEL_PATH = "best-512_ncnn_model"  # model in use

    # MODEL_PATH = "yolov8n.pt"  # lightweight YOLOv8 Nano model
    DETECTION_IMAGE_SIZE = 512
    CONFIDENCE_THRESHOLD = 0.4
    MAX_DETECTIONS = 15
    VIDEO_INPUT_PATH = "video_source/pending/17.mp4"
    PRIMARY_LINE_POSITION = 4 / 6
    SECONDARY_LINE_POSITION = 1 / 6
    COUNTING_ZONE_MARGIN = 70
    FORKLIFT_CLASS_NAME = "forklift"
    BALE_CLASS_NAME = "akhal"
    SHOW_DISPLAY = True
    DISPLAY_WINDOW_NAME = "Bale Counter"
    PROCESS_EVERY_N_FRAMES = 3
    SAVE_SNAPSHOTS = True


# ====================== SIMPLE TRACKER ======================
class SimpleTracker:
    """A simple lightweight tracker for Raspberry Pi"""
    
    def __init__(self, primary_line_x, secondary_line_x, config):
        self.primary_line_x = primary_line_x
        self.secondary_line_x = secondary_line_x
        self.margin = config.COUNTING_ZONE_MARGIN
        self.forklift_class = config.FORKLIFT_CLASS_NAME.lower()
        self.bale_class = config.BALE_CLASS_NAME.lower()
        
        self.objects = {}
        self.next_id = 1
        self.primary_count = 0
        self.secondary_count = 0
        self.forklift_entered = False
        self.last_detection_time = {}
        self.max_age = 10  # maximum missed frames
        
    def update(self, detections, frame_num):
        current_time = time.time()
        active_ids = []
        
        # Mark old objects as inactive
        for obj_id in list(self.objects.keys()):
            if current_time - self.last_detection_time.get(obj_id, 0) > 0.5:  # 0.5 seconds
                self.objects[obj_id]['active'] = False
        
        # Match and update
        for det in detections:
            cx, cy = det['cx'], det['cy']
            matched_id = self._find_match(cx, cy)
            
            if matched_id is None:
                # New object
                matched_id = self.next_id
                self.objects[matched_id] = {
                    'id': matched_id,
                    'cx': cx,
                    'cy': cy,
                    'prev_cx': cx,
                    'class': det['class_name'],
                    'counted_primary': False,
                    'counted_secondary': False,
                    'active': True,
                    'bbox': det['bbox'],
                    'conf': det['conf']
                }
                self.next_id += 1
            else:
                # Update existing object
                obj = self.objects[matched_id]
                obj['prev_cx'] = obj['cx']
                obj['cx'] = cx
                obj['cy'] = cy
                obj['class'] = det['class_name']
                obj['bbox'] = det['bbox']
                obj['conf'] = det['conf']
                obj['active'] = True
            
            self.last_detection_time[matched_id] = current_time
            active_ids.append(matched_id)
            
            # Check line crossing
            self._check_crossing(matched_id)
        
        # Remove old inactive objects
        to_delete = []
        for obj_id, obj in self.objects.items():
            if obj_id not in active_ids and not obj['active']:
                to_delete.append(obj_id)
        
        for obj_id in to_delete:
            del self.objects[obj_id]
            if obj_id in self.last_detection_time:
                del self.last_detection_time[obj_id]
        
        # Prepare output
        tracks = []
        for obj_id in active_ids:
            obj = self.objects[obj_id]
            tracks.append({
                'track_id': obj_id,
                'bbox': obj['bbox'],
                'cx': obj['cx'],
                'cy': obj['cy'],
                'class_name': obj['class'],
                'conf': obj['conf'],
                'counted_primary': obj['counted_primary'],
                'counted_secondary': obj['counted_secondary']
            })
        
        return tracks
    
    def _find_match(self, cx, cy, max_distance=50):
        """Find the nearest existing object"""
        best_id = None
        best_dist = max_distance
        
        for obj_id, obj in self.objects.items():
            if not obj['active']:
                continue
                
            dist = np.sqrt((cx - obj['cx'])**2 + (cy - obj['cy'])**2)
            if dist < best_dist:
                best_dist = dist
                best_id = obj_id
        
        return best_id
    
    def _check_crossing(self, obj_id):
        """Check crossing of counting lines"""
        obj = self.objects[obj_id]
        
        # Calculate movement direction
        moved_right = obj['cx'] > obj['prev_cx']
        moved_left = obj['cx'] < obj['prev_cx']
        
        # Crossing first line (entry)
        if (not obj['counted_primary'] and 
            obj['prev_cx'] > self.primary_line_x and 
            obj['cx'] <= self.primary_line_x and
            abs(obj['cx'] - self.primary_line_x) <= self.margin):
            
            obj['counted_primary'] = True
            self.primary_count += 1
            
            if obj['class'] == self.forklift_class:
                self.forklift_entered = True
            elif obj['class'] == self.bale_class:
                print(f"  [COUNT] Bale entered! Total: {self.primary_count}")
        
        # Crossing second line (exit)
        if (not obj['counted_secondary'] and 
            obj['prev_cx'] > self.secondary_line_x and 
            obj['cx'] <= self.secondary_line_x and
            abs(obj['cx'] - self.secondary_line_x) <= self.margin):
            
            obj['counted_secondary'] = True
            self.secondary_count += 1
            
            if obj['class'] == self.bale_class:
                print(f"  [COUNT] Bale exited! Total: {self.secondary_count}")


# ====================== DETECTOR ======================
try:
    from ultralytics import YOLO
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Using dummy detector.")

class ObjectDetector:
    def __init__(self, config):
        self.config = config
        self.imgsz = config.DETECTION_IMAGE_SIZE
        self.conf_thresh = config.CONFIDENCE_THRESHOLD
        
        if TORCH_AVAILABLE:
            print(f"Loading model: {config.MODEL_PATH}")
            try:
                # Load model with minimal settings
                self.model = YOLO(config.MODEL_PATH)
                print("Model loaded successfully")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Falling back to dummy detector")
                self.model = None
        else:
            self.model = None
    
    def detect(self, frame):
        if self.model is None or not TORCH_AVAILABLE:
            return []
        
        try:
            # Run detection with minimal parameters
            results = self.model(
                frame, 
                imgsz=self.imgsz,
                conf=self.conf_thresh,
                max_det=self.config.MAX_DETECTIONS,
                verbose=False,
                device='cpu'  # force CPU usage
            )
            
            detections = []
            if results and len(results) > 0:
                result = results[0]
                
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    
                    for i in range(len(boxes)):
                        box = boxes.xyxy[i].cpu().numpy()
                        conf = boxes.conf[i].cpu().numpy()
                        cls_id = int(boxes.cls[i].cpu().numpy())
                        
                        # Class name
                        try:
                            class_name = self.model.names[cls_id].lower()
                        except:
                            class_name = f"class_{cls_id}"
                        
                        # Fix bale/akhal naming issue
                        if 'bale' in class_name or 'akhal' in class_name:
                            class_name = 'akhal'
                        elif 'fork' in class_name or 'forklift' in class_name:
                            class_name = 'forklift'
                        
                        x1, y1, x2, y2 = map(int, box)
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        
                        detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'cx': cx,
                            'cy': cy,
                            'class_name': class_name,
                            'conf': float(conf)
                        })
            
            return detections
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []


# ====================== MAIN APPLICATION ======================
class BaleCounter:
    def __init__(self, config):
        self.config = config
        
        # Open video
        print(f"Opening video: {config.VIDEO_INPUT_PATH}")
        self.cap = cv2.VideoCapture(config.VIDEO_INPUT_PATH)
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {config.VIDEO_INPUT_PATH}")
        
        # Video information
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video info: {self.width}x{self.height}, {self.fps:.1f} FPS, {self.total_frames} frames")
        
        # Calculate lines
        self.primary_x = int(self.width * config.PRIMARY_LINE_POSITION)
        self.secondary_x = int(self.width * config.SECONDARY_LINE_POSITION)
        
        print(f"Primary line: x={self.primary_x}")
        print(f"Secondary line: x={self.secondary_x}")
        
        # Create components
        self.detector = ObjectDetector(config)
        self.tracker = SimpleTracker(self.primary_x, self.secondary_x, config)
        
        # State variables
        self.frame_count = 0
        self.processed_count = 0
        self.start_time = None
        self.last_counts = {'primary': 0, 'secondary': 0}
        
        # Create directories
        os.makedirs('results', exist_ok=True)
        os.makedirs('snapshots', exist_ok=True)
    
    def run(self):
        print("\n" + "="*50)
        print("STARTING BALE COUNTER")
        print("="*50)
        
        self.start_time = time.time()
        fps_buffer = deque(maxlen=30)
        
        try:
            while True:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("\nEnd of video reached")
                    break
                
                self.frame_count += 1
                
                # Skip frames based on config
                if self.frame_count % self.config.PROCESS_EVERY_N_FRAMES != 0:
                    # Show frame without processing
                    if self.config.SHOW_DISPLAY:
                        self._show_frame(frame, skip=True)
                    continue
                
                self.processed_count += 1
                
                # Object detection
                detect_start = time.time()
                detections = self.detector.detect(frame)
                detect_time = time.time() - detect_start
                
                # Tracking
                tracks = self.tracker.update(detections, self.frame_count)
                
                # Calculate FPS
                fps_buffer.append(time.time())
                if len(fps_buffer) > 1:
                    fps = len(fps_buffer) / (fps_buffer[-1] - fps_buffer[0])
                else:
                    fps = 0
                
                # Draw on frame
                annotated = self._annotate_frame(frame.copy(), tracks, detect_time, fps)
                
                # Save snapshot when count changes
                self._save_snapshot_if_needed(annotated)
                
                # Display
                if self.config.SHOW_DISPLAY:
                    self._show_frame(annotated)
                
                # Show progress
                if self.processed_count % 5 == 0:
                    self._print_progress(fps, detect_time)
                
                # Check exit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nUser requested exit")
                    break
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._cleanup()
    
    def _annotate_frame(self, frame, tracks, detect_time, fps):
        """Draw on frame"""
        
        # Draw counting lines
        cv2.line(frame, (self.primary_x, 0), (self.primary_x, self.height), 
                (0, 0, 255), 2)
        cv2.line(frame, (self.secondary_x, 0), (self.secondary_x, self.height), 
                (0, 255, 255), 2)
        
        # Draw line margins
        margin = self.config.COUNTING_ZONE_MARGIN
        cv2.line(frame, (self.primary_x - margin, 0), 
                (self.primary_x - margin, self.height), (0, 0, 255), 1)
        cv2.line(frame, (self.primary_x + margin, 0), 
                (self.primary_x + margin, self.height), (0, 0, 255), 1)
        
        # Draw objects
        for track in tracks:
            x1, y1, x2, y2 = track['bbox']
            
            # Select color
            if track['class_name'] == 'forklift':
                color = (255, 0, 0)  # blue
            elif track['class_name'] == 'akhal':
                color = (0, 255, 0)  # green
            else:
                color = (255, 255, 0)  # yellow
            
            if track['counted_primary'] or track['counted_secondary']:
                color = (0, 255, 255)  # light yellow for counted objects
            
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Write ID and class
            label = f"{track['track_id']}:{track['class_name'][:3]}"
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Write info
        info_lines = [
            f"Frame: {self.frame_count}/{self.total_frames}",
            f"FPS: {fps:.1f}",
            f"Detect: {detect_time*1000:.1f}ms",
            f"Primary: {self.tracker.primary_count}",
            f"Secondary: {self.tracker.secondary_count}",
            f"Total: {max(self.tracker.primary_count, self.tracker.secondary_count)}",
            f"Objects: {len(tracks)}"
        ]
        
        for i, line in enumerate(info_lines):
            y_pos = 30 + i * 25
            color = (255, 255, 255)
            
            if "Primary" in line:
                color = (0, 0, 255)
            elif "Secondary" in line:
                color = (0, 255, 255)
            elif "Total" in line:
                color = (0, 255, 0)
            
            cv2.putText(frame, line, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame
    
    def _show_frame(self, frame, skip=False):
        """Display frame"""
        if skip:
            # Show resized frame
            small = cv2.resize(frame, (640, 360))
            cv2.imshow(self.config.DISPLAY_WINDOW_NAME + " [SKIP]", small)
        else:
            small = cv2.resize(frame, (640, 360))
            cv2.imshow(self.config.DISPLAY_WINDOW_NAME, small)
    
    def _print_progress(self, fps, detect_time):
        """Display progress"""
        elapsed = time.time() - self.start_time
        if self.total_frames > 0:
            progress = (self.frame_count / self.total_frames) * 100
        else:
            progress = 0
        
        print(f"Frame: {self.frame_count:5d} ({progress:5.1f}%) | "
              f"FPS: {fps:5.1f} | "
              f"Detect: {detect_time*1000:5.1f}ms | "
              f"In: {self.tracker.primary_count:3d} | "
              f"Out: {self.tracker.secondary_count:3d}")
    
    def _save_snapshot_if_needed(self, frame):
        """Save snapshot when count changes"""
        current_counts = {
            'primary': self.tracker.primary_count,
            'secondary': self.tracker.secondary_count
        }
        
        # Check whether count changed
        if (current_counts['primary'] > self.last_counts['primary'] or
            current_counts['secondary'] > self.last_counts['secondary']):
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"snapshot_{self.frame_count}_{timestamp}.jpg"
            filepath = os.path.join('snapshots', filename)
            
            cv2.imwrite(filepath, frame)
            print(f"  [SNAPSHOT] Count changed! Saved to {filename}")
            
            # Save event
            event = {
                'timestamp': time.time(),
                'frame': self.frame_count,
                'counts': current_counts.copy(),
                'filename': filename
            }
            
            event_file = os.path.join('results', 'events.jsonl')
            with open(event_file, 'a') as f:
                f.write(json.dumps(event) + '\n')
        
        self.last_counts = current_counts.copy()
    
    def _cleanup(self):
        """Clean up resources"""
        elapsed = time.time() - self.start_time
        
        print("\n" + "="*50)
        print("PROCESSING COMPLETE")
        print("="*50)
        
        if elapsed > 0:
            print(f"Total time: {elapsed:.1f} seconds")
            print(f"Frames read: {self.frame_count}")
            print(f"Frames processed: {self.processed_count}")
            print(f"Average FPS: {self.frame_count/elapsed:.1f}")
        
        print(f"Primary count (in): {self.tracker.primary_count}")
        print(f"Secondary count (out): {self.tracker.secondary_count}")
        print(f"Total count: {max(self.tracker.primary_count, self.tracker.secondary_count)}")
        
        # Save final report
        report = {
            'video': self.config.VIDEO_INPUT_PATH,
            'model': self.config.MODEL_PATH,
            'start_time': self.start_time,
            'end_time': time.time(),
            'duration': elapsed,
            'frames_total': self.frame_count,
            'frames_processed': self.processed_count,
            'primary_count': self.tracker.primary_count,
            'secondary_count': self.tracker.secondary_count,
            'total_count': max(self.tracker.primary_count, self.tracker.secondary_count),
            'config': {
                'detection_size': self.config.DETECTION_IMAGE_SIZE,
                'confidence': self.config.CONFIDENCE_THRESHOLD,
                'frame_skip': self.config.PROCESS_EVERY_N_FRAMES
            }
        }
        
        report_file = os.path.join('results', 'final_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nReport saved to: {report_file}")
        
        # Release resources
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # If PyTorch is used, clear GPU memory
        if TORCH_AVAILABLE:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print("Resources cleaned up successfully")
        print("="*50)


# ====================== MAIN ======================
if __name__ == "__main__":
    # First, check whether the video exists
    config = Config()
    
    if not os.path.exists(config.VIDEO_INPUT_PATH):
        print(f"ERROR: Video file not found: {config.VIDEO_INPUT_PATH}")
        print("Please update VIDEO_INPUT_PATH in Config class")
        exit(1)
    
    # Check model
    if not os.path.exists(config.MODEL_PATH):
        print(f"WARNING: Model file not found: {config.MODEL_PATH}")
        print("Downloading a lightweight YOLOv8n model...")
        try:
            from ultralytics import YOLO
            model = YOLO('weights/best.pt')
            config.MODEL_PATH = 'weights/best.pt'
            print("Model downloaded successfully")
        except:
            print("Could not download model. Using dummy detector.")
    
    # Run application
    app = BaleCounter(config)
    app.run()