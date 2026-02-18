"""
Video Reader Module

Handles reading frames from video file or PiCamera2 in a separate thread.
"""

import cv2
import threading
import queue
import time
import os

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False

# PICAMERA_MAIN_SIZE = (1280, 720)
# PICAMERA_SENSOR_SIZE = (1280, 720)
PICAMERA_MAIN_SIZE = (1960, 1080)
PICAMERA_SENSOR_SIZE = (1960, 1080)


def _get_env_float(name):
    val = os.getenv(name)
    if not val:
        return None
    try:
        return float(val)
    except ValueError:
        return None

class VideoReader:
    """Thread-safe video reader class."""

    def __init__(self, video_path, queue_size=128):
        """
        Args:
            video_path: Path to input video file or the string "picamera2".
            queue_size: Maximum size of frame queue.
        """
        self.video_path = video_path
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.thread = None
        self.cap = None
        self.picam2 = None
        self.width = 0
        self.height = 0
        self.fps = 0.0
        self.total_frames = 0
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
                main={"size": PICAMERA_MAIN_SIZE, "format": "RGB888"},
                sensor={"output_size": PICAMERA_SENSOR_SIZE, "bit_depth": 10}
            )
            config["raw"]["size"] = PICAMERA_SENSOR_SIZE
            self.picam2.configure(config)
            self.picam2.start()
            exposure_us = _get_env_float("ANBAR_EXPOSURE_US")
            analogue_gain = _get_env_float("ANBAR_ANALOG_GAIN")
            digital_gain = _get_env_float("ANBAR_DIGITAL_GAIN")
            if exposure_us or analogue_gain or digital_gain:
                controls = {"AeEnable": False}
                if exposure_us:
                    controls["ExposureTime"] = int(exposure_us)
                if analogue_gain:
                    controls["AnalogueGain"] = analogue_gain
                if digital_gain:
                    controls["DigitalGain"] = digital_gain
                self.picam2.set_controls(controls)
            time.sleep(2)
            self.width = config["main"]["size"][0]
            self.height = config["main"]["size"][1]
            self.fps = 30.0
            self.total_frames = 0
        except Exception as exc:
            print(f"PiCamera2 initialization failed: {exc}")
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
                self.frame_queue.put({
                    "frame": frame,
                    "frame_number": frame_number,
                    "timestamp": time.time()
                })
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
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "total_frames": self.total_frames
        }

