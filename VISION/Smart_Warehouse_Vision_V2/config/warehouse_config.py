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
    WEIGHTS_DEFAULT = '100_epoch_650_yolov11n_v2.pt'  
    
    PALLETE_CLASS_MAP = {
        1: "sulfat",
        2: "pack",
        3: "neshaste",
        4: "paper-roll"
    }

    VALID_CLASSES = set(PALLETE_CLASS_MAP.keys()) 

    MODEL_INPUT_SIZE_DEFAULT = 256
    FRAME_SKIP_DEFAULT = 1
    IOU_THRESH_DEFAULT = 0.5
    CONF_THRESH_DEFAULT = 0.65
    
    IOU_THRESHOLD = 0.3  
    MIN_DETECTION_CONFIDENCE = 0.3      
    MIN_HITS_TO_CONFIRM = 3  
    MAX_MISSES = 15  
    TRACK_HISTORY_LEN = 20              
    COUNTING_COOLDOWN_SECONDS = 1.5     

    COUNTING_ZONE_X_START_RATIO = 0.385
    COUNTING_ZONE_Y_START_RATIO = 0.040
    COUNTING_ZONE_X_END_RATIO = 0.757
    COUNTING_ZONE_Y_END_RATIO = 1.0

    EVENT_COOLDOWN_SECONDS = 5
    
    # System Configuration
    TIMEZONE = ZoneInfo('Asia/Tehran')

    # VALID_CLASSES = {0, 1, 2, 3}  # Adjust if needed