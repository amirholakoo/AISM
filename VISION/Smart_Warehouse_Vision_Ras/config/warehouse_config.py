#config/warehouse_config.py
"""
Warehouse Configuration Module
Contains all configuration constants and settings for the Smart Warehouse CV System.
"""

from zoneinfo import ZoneInfo


class WarehouseConfig:
    """Configuration class containing all system constants and settings."""
    
    # Camera Configuration
    CAMERA_MAP = {
        "0": "Production Warehouse A",
        "1": "Heavy Warehouse B", 
        "2": "Packing Warehouse C",
        "3": "Storage Warehouse D",
    }
    
    # Model Configuration
    WEIGHTS_DEFAULT = 'final_yolov11n_640_100epoch.pt'  # Ensure you have this weights file
    
    # --- Class and UI Configuration ---
    # Mapping from class IDs to human-readable names
    # Class IDs should correspond to your trained YOLO model
    PALLETE_CLASS_MAP = {
        1: "sulfat",
        2: "pack",
        3: "neshaste",
        4: "paper-roll"
    }
    
    # Dynamically generate the set of valid classes from the map keys
    VALID_CLASSES = set(PALLETE_CLASS_MAP.keys()) # Classes to be detected and tracked

    # Detection Parameters
    MODEL_INPUT_SIZE_DEFAULT = 256
    FRAME_SKIP_DEFAULT = 1
    IOU_THRESH_DEFAULT = 0.5
    CONF_THRESH_DEFAULT = 0.65
    
    # Tracking and Counting Parameters
    IOU_THRESHOLD = 0.3  # IoU threshold for matching tracks with detections
    MIN_DETECTION_CONFIDENCE = 0.3      # Min confidence to create a new track
    MIN_HITS_TO_CONFIRM = 3  # Frames a track must exist to be 'CONFIRMED'
    MAX_MISSES = 15  # Frames a track can be 'COASTING' before deletion
    TRACK_HISTORY_LEN = 20              # Max length of track history
    COUNTING_COOLDOWN_SECONDS = 1.5     # Seconds a track must be gone to be counted

    # Region-Based Counting Configuration
    # Defines a central counting zone to prevent spurious counts from a single line.
    # An object must fully pass through this zone to be counted.
    # Values are proportions of the frame's width and height.
    COUNTING_ZONE_X_START_RATIO = 0.385
    COUNTING_ZONE_Y_START_RATIO = 0.040
    COUNTING_ZONE_X_END_RATIO = 0.757
    COUNTING_ZONE_Y_END_RATIO = 1.0
    
    # Event Cooldown
    # Prevents duplicate events for the same track within a short time frame.
    EVENT_COOLDOWN_SECONDS = 5
    
    # System Configuration
    TIMEZONE = ZoneInfo('Asia/Tehran')
    
    # This is now generated automatically from the PALLETE_CLASS_MAP
    # VALID_CLASSES = {0, 1, 2, 3}  # Adjust if needed