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
    WEIGHTS_DEFAULT = 'pallete_yolov8n.pt'  # Ensure you have this weights file
    
    # New Pallete Model Class Definitions
    # Assuming class IDs are 0: empty, 1: sulfat, 2: pack_material, 3: neshaste
    PALLETE_CLASS_MAP = {
        0: "empty_forklift",
        1: "sulfat",
        2: "pack_material",
        3: "neshaste"
    }
    LOADED_PALLETE_CLASSES = {1, 2, 3}  # Classes representing a loaded forklift
    EMPTY_PALLETE_CLASS = 0             # Class representing an empty forklift

    # Detection Parameters
    RESOLUTION_OPTIONS = {
        "Low (640x640)": 640,
        "Medium (960x960)": 960,
        "High (1280x1280)": 1280,
    }
    MODEL_INPUT_SIZE_DEFAULT = 1289
    LINE_X_DEFAULT = 900
    FRAME_SKIP_DEFAULT = 3
    IOU_THRESH_DEFAULT = 0.3
    CONF_THRESH_DEFAULT = 0.4
    
    # Tracking and Counting Parameters
    IOU_THRESHOLD = 0.3  # IoU threshold for matching tracks with detections
    MIN_DETECTION_CONFIDENCE = 0.4  # Minimum confidence to create a new track
    MIN_HITS_TO_CONFIRM = 3  # Frames a track must exist to be 'CONFIRMED'
    MAX_MISSES = 15  # Frames a track can be 'COASTING' before deletion
    TRACK_HISTORY_LEN = 30  # Max length of center point history for a track

    # Region-Based Counting Configuration
    # Defines a central counting zone to prevent spurious counts from a single line.
    # An object must fully pass through this zone to be counted.
    # Values are proportions of the frame's width and height.
    COUNTING_ZONE_X_START_RATIO = 0.385
    COUNTING_ZONE_Y_START_RATIO = 0.240
    COUNTING_ZONE_X_END_RATIO = 0.557
    COUNTING_ZONE_Y_END_RATIO = 1.0
    
    # Event Cooldown
    # Prevents duplicate events for the same track within a short time frame.
    EVENT_COOLDOWN_SECONDS = 5
    
    # System Configuration
    TIMEZONE = ZoneInfo('Asia/Tehran')
    VALID_CLASSES = {0, 1, 2, 3}  # Adjust if needed