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