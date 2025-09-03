import pytz

class QRScannerConfig:
    """Configuration parameters for the QR Scanner."""
    # Video Stream settings
    RECONNECT_DELAY_SECONDS = 5
    CAMERA_RESOLUTION = (1280, 960)
    FRAME_RATE = 15
    
    # Cache settings
    CONSOLE_LOG_TIMEOUT = 5.0

    # Performance settings
    DISPLAY_SCALE = 0.5

    # Output settings
    OUTPUT_DIR = "output"
    TIMEZONE = pytz.timezone('Asia/Tehran') 


    VALIDATION_FRAME_COUNT = 5
    STABILITY_PIXEL_THRESHOLD = 50

    FOCAL_LENGTH = 1639.43


    KNOWN_QR_CODE_WIDTH_CM = 15.0 
    MAX_SCAN_DISTANCE_CM = 100