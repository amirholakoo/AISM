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
    WEIGHTS_DEFAULT = 'best_yolov11n.pt'  # Ensure you have this weights file
    
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
    MODEL_INPUT_SIZE_DEFAULT = 640
    LINE_X_DEFAULT = 900
    FRAME_SKIP_DEFAULT = 6
    IOU_THRESH_DEFAULT = 0.3
    CONF_THRESH_DEFAULT = 0.4
    EVENT_COOLDOWN_SECONDS = 5  # New: Cooldown in seconds between events for the same track
    
    # System Configuration
    TIMEZONE = ZoneInfo('Asia/Tehran')
    VALID_CLASSES = {0, 1, 2, 3}  # Adjust if needed