import os

"""
Configuration file for Akhal Counting System
All configurable parameters are defined here
"""


class Config:
    """Configuration class for the counting system"""

    # Base Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RESULT_DIR = os.path.join(BASE_DIR, "result")
    TEMP_DIR = os.path.join(BASE_DIR, "temp")
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    PROCESSED_DIR = os.path.join(OUTPUT_DIR, "processed")
    LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
    LOG_FILE_PATH = os.path.join(BASE_DIR, "log.txt")

    # Model Configuration
    # MODEL_FILENAME = "best_50_new_ncnn_model_with_auggmentation_v1"
    # MODEL_FILENAME = "best_70_new_ncnn_model_with_auggmentation"
    MODEL_FILENAME = "weights/best_50_v1.onnx"
    # MODEL_FILENAME = "best-50-new-yolov11_ncnn_model"
    MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)
    DETECTION_IMAGE_SIZE = 512
    FRAME_SKIP = 2
    CONFIDENCE_THRESHOLD = 0.6
    MAX_DETECTIONS = 30

    # Video Configuration
    # VIDEO_SOURCE = "picamera2"
    VIDEO_SOURCE = "video_source/pending/13.mp4"  # or "picamera2"
    VIDEO_OUTPUT_NAME = "output.mp4"
    VIDEO_CODEC = "mp4v"
    PICAMERA_MAIN_SIZE = (1640, 1232)             # 4:3 field of view for IMX219
    PICAMERA_MAIN_FORMAT = "RGB888"
    PICAMERA_SENSOR_CONFIG = None                 # e.g. {"output_size": (1640, 1232), "bit_depth": 10}
    PICAMERA_STARTUP_DELAY = 2.0

    # Counting Line Configuration
    COUNTING_LINE_POSITION = 0.3
    COUNTING_ZONE_MARGIN = 45

    # Object Classes
    FORKLIFT_CLASS_NAME = "forklift"
    EXCLUDE_CLASSES = []

    # Tracking and Matching
    MATCH_DISTANCE = 50
    MAX_MISSED_FRAMES = 15
    TRACKING_PERSISTENCE = 20

    # Display / Output Configuration
    SHOW_DISPLAY = True
    DISPLAY_WINDOW_NAME = "Forklift Counter"
    DRAW_BOXES = True
    ENABLE_OUTPUT_VIDEO = False
    ENABLE_RUN_LOG = False
    DISPLAY_MAX_WIDTH = 960
    DISPLAY_MAX_HEIGHT = 540

    # Performance Configuration
    QUEUE_SIZE = 32
    FPS_SMOOTHING_FRAMES = 25
    PROCESS_EVERY_N_FRAMES = 2
    DETECTION_PIPELINE_DEPTH = 2
    DETECTION_WORKERS = 2

    # Visualization Colors (BGR format)
    COLOR_BBOX = (0, 255, 0)
    COLOR_COUNTED_OBJECT = (0, 255, 255)

    # Text Configuration
    FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE_LARGE = 0.9
    FONT_SCALE_MEDIUM = 0.7
    FONT_SCALE_SMALL = 0.5
    FONT_THICKNESS = 2

    # Logging Configuration
    VERBOSE = True

    # CPU / Threading Configuration
    OMP_NUM_THREADS = "4"
    MKL_NUM_THREADS = "4"
    OMP_DYNAMIC = "FALSE"
    KMP_AFFINITY = "granularity=fine,compact,1,0"

