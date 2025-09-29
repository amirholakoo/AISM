import torch
import cv2
import numpy as np
from picamera2 import Picamera2
import time
from ultralytics import YOLO
import os

# Set display environment for GUI
if 'DISPLAY' not in os.environ:
    os.environ['DISPLAY'] = ':0'
    os.environ['QT_QPA_PLATFORM'] = 'xcb'

class SackDetector:
    
    #def __init__(self, model_path='weights/yolov8n-obb-v4.pt', headless=False):
    def __init__(self, model_path='weights/best.pt', headless=False):

    #def __init__(self, model_path='yoloe-v11n.pt'):
    #def __init__(self, model_path='yoloe-v11n.onnx'):

        """
        Initialize the sack detector with YOLO model
        Args:
            model_path: Path to the YOLO model weights
            headless: If True, run without GUI display
        """
        self.headless = headless
        # Initialize PiCamera
        self.camera = Picamera2()
        
        # Configure camera for lower resolution for faster processing
        config = self.camera.create_preview_configuration(
            main={"size": (960, 720), "format": "RGB888"},
            raw={"size": (960, 720)} #1280,960
        )
        self.camera.configure(config)
        
        # Load YOLO model using Ultralytics YOLO class
        print("loading model")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = YOLO(model_path)
        print("model loaded")
        
        # Debug: Print model info
        print(f"Model file: {model_path}")
        print(f"Model task: {self.model.task}")
        print(f"Model names: {self.model.names}")
        print(f"Model device: {self.model.device}")
        
        # Test model with dummy input
        print("Testing model with dummy input...")
        dummy_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        test_results = self.model(dummy_input, conf=0.5, verbose=False)#0.1
        print(f"Dummy test results: {len(test_results)} result objects")
        for i, result in enumerate(test_results):
            print(f"  Test result {i} has obb: {hasattr(result, 'obb')}")
            print(f"  Test result {i} has boxes: {hasattr(result, 'boxes')}")
        
        # For ONNX models, device is specified during inference, not model loading
        
        # Set confidence threshold - lowered for debugging
        self.conf_threshold = 0.5
        
        # For FPS calculation
        self.prev_time = 0
        self.frame_count = 0
        self.fps = 0
        self.start_time = time.time()
        
    def detect(self, frame):
        """
        Perform detection on a frame using YOLOv8 OBB model
        """
        # Debug: Print frame info
        print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")
        print(f"Frame min/max values: {frame.min()}/{frame.max()}")
        
        # The model expects RGB, which is what Picamera2 provides.
        # No color conversion is needed before passing the frame to the model.
        rgb_frame = frame
        
        # Perform inference using Ultralytics YOLO
        device = 0 if self.device.type == 'cuda' else 'cpu'
        print(f"Running inference on device: {device}")
        print(f"Confidence threshold: {self.conf_threshold}")
        
        results = self.model(rgb_frame, conf=self.conf_threshold, device=device, verbose=True)
        
        # Debug: Print results structure
        print(f"Number of result objects: {len(results)}")
        for i, result in enumerate(results):
            print(f"Result {i} attributes: {[attr for attr in dir(result) if not attr.startswith('_')]}")
            print(f"Result {i} has obb: {hasattr(result, 'obb')}")
            print(f"Result {i} has boxes: {hasattr(result, 'boxes')}")
            if hasattr(result, 'obb'):
                print(f"Result {i} obb is None: {result.obb is None}")
                if result.obb is not None:
                    print(f"Result {i} obb length: {len(result.obb)}")
            if hasattr(result, 'boxes'):
                print(f"Result {i} boxes is None: {result.boxes is None}")
                if result.boxes is not None:
                    print(f"Result {i} boxes length: {len(result.boxes)}")
        
        # Process predictions for OBB (Oriented Bounding Box) model
        detections = []
        for result in results:
            # For OBB models, use obb instead of boxes
            if hasattr(result, 'obb') and result.obb is not None and len(result.obb) > 0:
                print(f"Processing {len(result.obb)} OBB detections")
                for i, obb in enumerate(result.obb):
                    try:
                        print(f"Processing OBB {i}")
                        # Get oriented bounding box coordinates (8 points)
                        xyxyxyxy = obb.xyxyxyxy[0].cpu().numpy()  # 8 points for oriented box
                        conf = obb.conf[0].cpu().numpy()
                        cls = obb.cls[0].cpu().numpy()
                        
                        print(f"OBB {i}: conf={conf}, cls={cls}")
                        print(f"OBB {i} points: {xyxyxyxy}")
                        
                        # Convert 8 points to bounding rectangle for display
                        x_coords = xyxyxyxy[::2]  # x coordinates
                        y_coords = xyxyxyxy[1::2]  # y coordinates
                        x1, x2 = float(np.min(x_coords)), float(np.max(x_coords))
                        y1, y2 = float(np.min(y_coords)), float(np.max(y_coords))
                        
                        detections.append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'obb_points': xyxyxyxy.tolist(),  # Store original 8 points
                            'confidence': float(conf),
                            'class': int(cls)
                        })
                    except Exception as e:
                        print(f"Error processing OBB {i}: {e}")
                        
            # Fallback to regular boxes if obb is not available
            elif hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                print(f"Processing {len(result.boxes)} regular box detections")
                for i, box in enumerate(result.boxes):
                    try:
                        print(f"Processing box {i}")
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = box.cls[0].cpu().numpy()
                        
                        print(f"Box {i}: conf={conf}, cls={cls}, bbox=[{x1}, {y1}, {x2}, {y2}]")
                        
                        detections.append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': float(conf),
                            'class': int(cls)
                        })
                    except Exception as e:
                        print(f"Error processing box {i}: {e}")
            else:
                print("No detections found in this result")
                
        print(f"Total detections found: {len(detections)}")
        return detections
    
    def draw_detections(self, frame, detections):
        """
        Draw detection boxes and FPS on frame
        """
        # Draw FPS counter
        cv2.putText(frame, f'FPS: {self.fps:.1f}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw detections
        for det in detections:
            conf = det['confidence']
            
            # Draw oriented bounding box if available, otherwise regular box
            if 'obb_points' in det:
                # Draw oriented bounding box using 8 points
                obb_points = np.array(det['obb_points'], dtype=np.int32)
                obb_points = obb_points.reshape((-1, 1, 2))
                cv2.polylines(frame, [obb_points], True, (0, 255, 0), 2)
                
                # Get center point for text
                center_x = int(np.mean(obb_points[:, 0, 0]))
                center_y = int(np.mean(obb_points[:, 0, 1]))
                cv2.putText(frame, f'{conf:.2f}', (center_x, center_y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                # Draw regular bounding box
                x1, y1, x2, y2 = [int(coord) for coord in det['bbox']]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{conf:.2f}', (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame
    
    def calculate_fps(self):
        """
        Calculate frames per second
        """
        self.frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # Update FPS every second
        if elapsed_time > 1:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = current_time
        
        return self.fps
    
    def run_detection(self):
        """
        Run continuous detection loop
        """
        print("Starting detection...")
        self.camera.start()
        
        try:
            frame_save_count = 0
            while True:
                # Capture frame in BGR format
                frame = self.camera.capture_array()
                
                # Save first few frames for inspection
                if frame_save_count < 3:
                    frame_filename = f"debug_frame_{frame_save_count}.png"
                    # Convert RGB to BGR for saving with cv2
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(frame_filename, frame_bgr)
                    print(f"Saved frame for inspection: {frame_filename}")
                    frame_save_count += 1
                
                # Perform detection
                detections = self.detect(frame)
                
                # Calculate FPS
                self.calculate_fps()
                
                # Draw detections and FPS
                display_frame = self.draw_detections(frame.copy(), detections)
                
                # Display frame only if not in headless mode
                if not self.headless:
                    cv2.imshow('Sack Detection', display_frame)
                    
                    # Break loop on 'q' press
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    # In headless mode, just print detection info
                    if detections:
                        print(f"Detected {len(detections)} objects")
                        for i, det in enumerate(detections):
                            print(f"  Object {i+1}: confidence={det['confidence']:.2f}, class={det['class']}")
                    
                    # Break loop after some time in headless mode (optional)
                    # You can remove this if you want it to run indefinitely
                    if self.frame_count > 100:  # Run for 100 frames then stop
                        break
                    
        except KeyboardInterrupt:
            print("Stopping detection...")
        
        finally:
            self.camera.stop()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    # Initialize detector with GUI display
    detector = SackDetector(headless=False)
    # Run detection loop
    detector.run_detection()
