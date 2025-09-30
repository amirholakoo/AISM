import torch
import cv2
import numpy as np
from picamera2 import Picamera2
import time
from ultralytics import YOLO
import os
import threading
import queue

# Set display environment for GUI
if 'DISPLAY' not in os.environ:
    os.environ['DISPLAY'] = ':0'
    os.environ['QT_QPA_PLATFORM'] = 'xcb'

class SackTracker:
    def __init__(self, model_path='weights/best.pt', headless=False):
        """
        Initialize the sack detector and tracker with YOLO model
        Args:
            model_path: Path to the YOLO model weights
            headless: If True, run without GUI display
        """
        self.headless = headless
        self.camera = Picamera2()
        
        config = self.camera.create_preview_configuration(
            main={"size": (960, 720), "format": "RGB888"},  # Reduced resolution for higher FPS
            raw={"size": (960, 720)}
        )
        self.camera.configure(config)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(model_path)
        
        self.conf_threshold = 0.25
        
        self.frame_count = 0
        self.fps = 0
        self.start_time = time.time()
        
        self.frame_queue = queue.Queue(maxsize=3)
        self.result_queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.last_annotated = None  # To store last processed frame for skipping
    
    def detection_thread(self):
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.5)
                results = self.model.track(frame, conf=self.conf_threshold, device=0 if self.device.type == 'cuda' else 'cpu', verbose=False, persist=True)
                self.result_queue.put(results)
            except queue.Empty:
                continue
    
    def run_detection(self):
        """
        Run continuous detection and tracking loop using YOLO's built-in tracker with multi-threading
        """
        print("Starting detection and tracking...")
        self.camera.start()
        
        threading.Thread(target=self.detection_thread, daemon=True).start()
        
        try:
            process_every = 2  # Process every 2nd frame to increase FPS
            local_frame_count = 0
            while True:
                frame = self.camera.capture_array()
                
                local_frame_count += 1
                if local_frame_count % process_every == 0:
                    try:
                        self.frame_queue.put_nowait(frame.copy())
                    except queue.Full:
                        pass
                
                # Get results if available
                try:
                    results = self.result_queue.get_nowait()
                    print(f"Detected {len(results[0].boxes) if results[0].boxes is not None else 0} objects")
                    
                    self.last_annotated = results[0].plot()
                except queue.Empty:
                    if self.last_annotated is not None:
                        annotated_frame = self.last_annotated.copy()
                    else:
                        annotated_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    annotated_frame = self.last_annotated
                
                self.calculate_fps()
                
                cv2.putText(annotated_frame, f'FPS: {self.fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if not self.headless:
                    cv2.imshow('Sack Detection and Tracking', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    num_objects = len(results[0].boxes.id) if 'results' in locals() and results[0].boxes is not None and results[0].boxes.id is not None else 0
                    print(f"FPS: {self.fps:.1f} | Number of tracked objects: {num_objects}")
                    if self.frame_count > 100:
                        break
                
        except KeyboardInterrupt:
            print("Stopping detection and tracking...")
        
        finally:
            self.stop_event.set()
            self.camera.stop()
            if not self.headless:
                cv2.destroyAllWindows()

    def calculate_fps(self):
        """
        Calculate frames per second
        """
        self.frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        if elapsed_time > 1:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = current_time
        
        return self.fps

if __name__ == '__main__':
    tracker = SackTracker(headless=False)
    tracker.run_detection()

# import torch
# import cv2
# import numpy as np
# from picamera2 import Picamera2
# import time
# from ultralytics import YOLO
# import os

# # Set display environment for GUI
# if 'DISPLAY' not in os.environ:
#     os.environ['DISPLAY'] = ':0'
#     os.environ['QT_QPA_PLATFORM'] = 'xcb'

# class SackTracker:
#     def __init__(self, model_path='weights/best.pt', headless=False):
#         """
#         Initialize the sack detector and tracker with YOLO model
#         Args:
#             model_path: Path to the YOLO model weights
#             headless: If True, run without GUI display
#         """
#         self.headless = headless
#         self.camera = Picamera2()
        
#         config = self.camera.create_preview_configuration(
#             main={"size": (960, 720), "format": "RGB888"},
#             raw={"size": (960, 720)}
#         )
#         self.camera.configure(config)
        
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model = YOLO(model_path)
        
#         self.conf_threshold = 0.25
        
#         self.frame_count = 0
#         self.fps = 0
#         self.start_time = time.time()
    
#     def run_detection(self):
#         """
#         Run continuous detection and tracking loop using YOLO's built-in tracker
#         """
#         print("Starting detection and tracking...")
#         self.camera.start()
        
#         try:
#             while True:
#                 frame = self.camera.capture_array()
                
#                 results = self.model.track(frame, conf=self.conf_threshold, device=0 if self.device.type == 'cuda' else 'cpu', verbose=False, persist=True)
                
#                 print(f"Detected {len(results[0].boxes) if results[0].boxes is not None else 0} objects")
                
#                 annotated_frame = results[0].plot()
                
#                 self.calculate_fps()
                
#                 cv2.putText(annotated_frame, f'FPS: {self.fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
#                 if not self.headless:
#                     cv2.imshow('Sack Detection and Tracking', annotated_frame)
#                     if cv2.waitKey(1) & 0xFF == ord('q'):
#                         break
#                 else:
#                     num_objects = len(results[0].boxes.id) if results[0].boxes is not None and results[0].boxes.id is not None else 0
#                     print(f"FPS: {self.fps:.1f} | Number of tracked objects: {num_objects}")
#                     if self.frame_count > 100:
#                         break
                
#         except KeyboardInterrupt:
#             print("Stopping detection and tracking...")
        
#         finally:
#             self.camera.stop()
#             if not self.headless:
#                 cv2.destroyAllWindows()

#     def calculate_fps(self):
#         """
#         Calculate frames per second
#         """
#         self.frame_count += 1
#         current_time = time.time()
#         elapsed_time = current_time - self.start_time
        
#         if elapsed_time > 1:
#             self.fps = self.frame_count / elapsed_time
#             self.frame_count = 0
#             self.start_time = current_time
        
#         return self.fps

# if __name__ == '__main__':
#     tracker = SackTracker(headless=False)
#     tracker.run_detection()
