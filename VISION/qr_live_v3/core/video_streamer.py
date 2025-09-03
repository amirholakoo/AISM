
import cv2
import time
import os
import threading
import logging
from queue import Queue, Empty, Full

# Get a logger for this module
logger = logging.getLogger(__name__)

# Conditionally import picamera2
try:
    from picamera2 import Picamera2
    IS_PICAMERA_AVAILABLE = True
    logger.info("picamera2 library found.")
except ImportError:
    IS_PICAMERA_AVAILABLE = False
    logger.info("picamera2 library not found, falling back to cv2.VideoCapture.")


class VideoStreamer:
    """
    Handles connecting to and reading from various video sources, including RTSP streams
    and the Raspberry Pi camera, in a separate thread. Features robust connection retries,
    stream verification, and a frame queue to prevent frame drops.
    """
    def __init__(self, source, config):
        self.source = source
        self.config = config
        self.video_source = None  # Will be cv2.VideoCapture or Picamera2 instance
        self.is_running = False
        self.frame_queue = Queue(maxsize=120)  # Increased queue size for stability
        self.grab_thread = None
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        logger.info(f"VideoStreamer initialized. IS_PICAMERA_AVAILABLE: {IS_PICAMERA_AVAILABLE}")

    def connect(self):
        """
        Initializes and connects to the specified video source.
        Returns True if connection is successful, False otherwise.
        """
        if self._initialize_video_source(self.source):
            self.is_running = True
            self.grab_thread = threading.Thread(target=self._grab_loop, daemon=True)
            self.grab_thread.start()
            return True
        return False

    def _initialize_video_source(self, source):
        """Initialize the correct video source based on input."""
        logger.info(f"Attempting to initialize video source: {source}")
        if source.lower() == "picamera" and IS_PICAMERA_AVAILABLE:
            try:
                logger.info("Initializing Raspberry Pi camera with new 'cam1' performance settings.")
                picam2 = Picamera2()

                script_dir = os.path.dirname(__file__)
                project_root = os.path.abspath(os.path.join(script_dir, '..'))
                tuning_file = os.path.join(project_root, 'imx219_noir.json')

                if os.path.exists(tuning_file):
                    picam2.load_tuning_file(tuning_file)
                    logger.info(f"Loaded custom tuning file: {tuning_file}")
                else:
                    logger.warning(f"Tuning file not found at {tuning_file}, using default tuning.")
                
                config = picam2.create_video_configuration(
                    main={"size": (1280, 960), "format": "RGB888"},
                    raw={'size': (1640, 1232)},
                    controls={
                        "FrameDurationLimits": (66666, 66666), # For 15 FPS
                        "ExposureTime": 8000,      # Corresponds to --shutter 8000
                        "AnalogueGain": 4.0        # Corresponds to --gain 4
                    }
                )
                picam2.configure(config)
                picam2.start()
                time.sleep(2.0)  # Allow camera to warm up
                self.video_source = picam2
                logger.info("✅ picamera2 started successfully with updated 'cam1' settings.")
                return True
            except Exception as e:
                logger.error(f"❌ Failed to initialize picamera2: {e}", exc_info=True)
                return False
        elif source.lower() == "picamera":
             logger.error("Pi Camera was selected, but the picamera2 library is not available.")
             return False
        else:
            return self._connect_opencv(source)

    def _connect_opencv(self, source):
        """Connect to a video source using OpenCV with retry logic."""
        logger.info(f"Initializing cv2.VideoCapture for source: {source}")
        max_retries = 8
        retry_delay = 2
        for attempt in range(max_retries):
            logger.info(f"Connection attempt {attempt + 1}/{max_retries} to {source}")
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            
            if cap and cap.isOpened():
                cap.set(cv2.CAP_PROP_FPS, self.config.FRAME_RATE)
                logger.info(f"Attempt {attempt + 1}: Source connected. Verifying stream...")
                grab_success = cap.grab()
                if grab_success:
                    logger.info(f"✅ Attempt {attempt + 1}: Test frame grabbed successfully. Stream is live.")
                    self.video_source = cap
                    self.video_source.set(cv2.CAP_PROP_BUFFERSIZE, 3)
                    return True
                else:
                    logger.warning(f"⚠️ Attempt {attempt + 1}: Connected but failed to grab frame.")
                    cap.release()
            else:
                logger.warning(f"⚠️ Attempt {attempt + 1}: Failed to open source.")
                if cap:
                    cap.release()

            time.sleep(retry_delay)
        
        logger.error(f"❌ Failed to open video source after {max_retries} attempts: {source}")
        return False

    def _grab_loop(self):
        """Continuously grabs frames and puts them in the queue."""
        logger.info("Frame grabber loop started.")
        while self.is_running:
            ret, frame = self._read_frame()
            if not ret:
                logger.warning("Failed to grab frame. Assuming stream ended. Stopping grabber.")
                try:
                    self.frame_queue.put(None, timeout=0.5)
                except Full:
                    pass # If full, processor will handle it
                break
            
            try:
                # This is a blocking call. If queue is full, it waits.
                self.frame_queue.put((frame, time.time()), timeout=1.0)
            except Full:
                logger.warning("Frame queue is full. Frame grabber is blocked. Processing might be slow.")
                continue
        logger.info("Frame grabber loop finished.")

    def _read_frame(self):
        """Reads a single frame from the initialized video source."""
        if IS_PICAMERA_AVAILABLE and isinstance(self.video_source, Picamera2):
            return True, self.video_source.capture_array()
        
        if isinstance(self.video_source, cv2.VideoCapture):
            ret = self.video_source.grab()
            if not ret:
                return False, None
            return self.video_source.retrieve()

        return False, None

    def read(self):
        """Reads the latest frame from the queue."""
        if not self.is_running:
            return False, None
        
        try:
            item = self.frame_queue.get(timeout=2.0)
            if item is None: # Sentinel for stream end
                self.is_running = False
                return False, None
            return True, item[0]
        except Empty:
            logger.warning("Frame queue was empty for 2 seconds.")
            return False, None

    def stop(self):
        """Stops the grabbing thread and releases the video capture object."""
        self.is_running = False
        if self.grab_thread is not None:
            self.grab_thread.join(timeout=2)
        
        if self.video_source:
            if IS_PICAMERA_AVAILABLE and isinstance(self.video_source, Picamera2):
                if self.video_source.started:
                    self.video_source.stop()
            elif isinstance(self.video_source, cv2.VideoCapture):
                self.video_source.release()
        
        logger.info("Video stream stopped.") 