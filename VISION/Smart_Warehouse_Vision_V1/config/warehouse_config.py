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
    WEIGHTS_DEFAULT = 'pallete_yolov8n.pt'  
    
    PALLETE_CLASS_MAP = {
        0: "empty_forklift",
        1: "sulfat",
        2: "pack_material",
        3: "neshaste"
    }
    LOADED_PALLETE_CLASSES = {1, 2, 3}  
    EMPTY_PALLETE_CLASS = 0            

    # Detection Parameters
    MODEL_INPUT_SIZE_DEFAULT = 1280  
    LINE_X_DEFAULT = 900
    FRAME_SKIP_DEFAULT = 3
    IOU_THRESH_DEFAULT = 0.3
    CONF_THRESH_DEFAULT = 0.3
    EVENT_COOLDOWN_SECONDS = 5 
    
    # System Configuration
    TIMEZONE = ZoneInfo('Asia/Tehran')
    VALID_CLASSES = {0, 1, 2, 3}  # Adjust if needed