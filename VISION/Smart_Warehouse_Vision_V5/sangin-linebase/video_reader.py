"""
Video Reader Module
Handles reading frames from video file in a separate thread
Safe version with lazy picamera2 import
"""

import cv2
import threading
import queue
import time
import os

print("video-reader-start")

# Don't import picamera2 at module level to avoid numpy errors
PICAMERA2_AVAILABLE = False

class VideoReader:
    """Thread-safe video reader class"""
    
    def __init__(self, video_path, queue_size=128, picamera_options=None):
        """
        Initialize video reader
        
        Args:
            video_path: Path to input video file or special value "picamera2"
            queue_size: Maximum size of frame queue
        """
        self.video_path = video_path
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.thread = None
        self.picamera_options = picamera_options or {}

        self.cap = None
        self.picam2 = None
        self.width = 0
        self.height = 0
        self.fps = 0
        self.total_frames = 0

        if video_path == "picamera2":
            self._init_picamera2()
        else:
            self._init_opencv_capture(video_path)
        
    def _init_opencv_capture(self, source):
        # If a file path is given and it does not exist, try PiCamera2
        if isinstance(source, str) and not source.isdigit() and not os.path.exists(source):
            print(f"{source} not found, trying PiCamera2")
            self._init_picamera2()
            if self.picam2 is not None:
                return

        self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            # If opening failed, also try PiCamera2
            print(f"Cannot open {source}, trying PiCamera2")
            if self.cap is not None:
                self.cap.release()
            self.cap = None
            self._init_picamera2()
            if self.picam2 is not None:
                return
            raise ValueError(f"Cannot open video file or camera: {source}")
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
    def _init_picamera2(self):
        """Initialize PiCamera2 with lazy import"""
        try:
            # Lazy import to avoid module-level errors
            from picamera2 import Picamera2
            
            # Try to load tuning file if it exists
            tuning_file = "imx219_noir.json"
            if os.path.exists(tuning_file):
                tuning = Picamera2.load_tuning_file(tuning_file)
                self.picam2 = Picamera2(tuning=tuning)
            else:
                print(f"Tuning file {tuning_file} not found, using default")
                self.picam2 = Picamera2()
            
            main_size = self.picamera_options.get("main_size", (1920, 1080))
            main_format = self.picamera_options.get("main_format", "RGB888")
            sensor_config = self.picamera_options.get("sensor_config")
            controls = self.picamera_options.get("controls")
            buffer_count = self.picamera_options.get("buffer_count")

            preview_kwargs = {
                "main": {"size": tuple(main_size), "format": main_format}
            }
            if sensor_config:
                preview_kwargs["sensor"] = sensor_config
            if controls:
                preview_kwargs["controls"] = controls
            if buffer_count:
                preview_kwargs["buffer_count"] = buffer_count

            config = self.picam2.create_preview_configuration(**preview_kwargs)
            
            self.picam2.configure(config)
            self.picam2.start()
            startup_delay = float(self.picamera_options.get("startup_delay", 2.0))
            if startup_delay > 0:
                time.sleep(startup_delay)

            self.width = config["main"]["size"][0]
            self.height = config["main"]["size"][1]
            self.fps = 30.0
            self.total_frames = 0
            print(f"PiCamera2 initialized: {self.width}x{self.height}")
            
        except Exception as e:
            print(f"PiCamera2 initialization failed: {e}")
            self.picam2 = None
            # Fallback to webcam
            print("Falling back to webcam (device 0)")
            self._init_opencv_capture(0)
        
    def start(self):
        """Start the video reading thread"""
        self.thread = threading.Thread(target=self._read_frames, daemon=True)
        self.thread.start()
        return self
    
    def _read_frames(self):
        """Internal method to read frames in a separate thread"""
        frame_number = 0
        
        while not self.stopped:
            if not self.frame_queue.full():
                if self.picam2 is not None:
                    try:
                        frame = self.picam2.capture_array()
                        ret = frame is not None
                    except Exception as e:
                        print(f"Error capturing from PiCamera2: {e}")
                        ret = False
                        frame = None
                else:
                    ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    self.stopped = True
                    break
                
                frame_number += 1
                
                self.frame_queue.put({
                    'frame': frame,
                    'frame_number': frame_number,
                    'timestamp': time.time()
                })
            else:
                time.sleep(0.001)
        
        if self.cap is not None:
            self.cap.release()
        if self.picam2 is not None:
            try:
                self.picam2.stop()
            except:
                pass
    
    def read(self):
        """
        Read next frame from queue
        
        Returns:
            Dictionary with frame data or None if stopped
        """
        try:
            return self.frame_queue.get(timeout=1.0)
        except queue.Empty:
            return None
    
    def more(self):
        """Check if there are more frames to read"""
        return not self.stopped or not self.frame_queue.empty()
    
    def stop(self):
        """Stop the video reader"""
        self.stopped = True
        if self.thread is not None:
            self.thread.join()
    
    def get_properties(self):
        """Get video properties"""
        return {
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'total_frames': self.total_frames
        }
