import logging
from logging.handlers import RotatingFileHandler
import sys

# Custom filter to only allow logs from the 'core' or '__main__' modules
class AppFilter(logging.Filter):
    def filter(self, record):
        # Also exclude special event logs meant only for the file
        return record.name.startswith('core') and not record.getMessage().startswith('[EVENT]')

# Custom filter to ONLY allow logs marked as [EVENT]
class EventFilter(logging.Filter):
    def filter(self, record):
        return record.getMessage().startswith('[EVENT]')

def setup_logging():
    """
    Configures logging for the application.
    - A rotating file log (`warehouse_events.log`) for critical events only (counts, snapshots).
    - A cleaner console log for real-time operational monitoring.
    """
    # Get the root logger
    root_logger = logging.getLogger()
    
    # Avoid adding handlers multiple times
    if root_logger.hasHandlers() and any(isinstance(h, RotatingFileHandler) for h in root_logger.handlers):
        return
        
    root_logger.setLevel(logging.INFO)

    # Clean up any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create a detailed formatter for the file log
    file_formatter = logging.Formatter(
        '%(asctime)s - %(message)s', # Simpler format for the event log
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Create a readable formatter for the console
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')

    # 1. Handler for warehouse_events.log (critical events ONLY)
    file_handler = RotatingFileHandler(
        'warehouse_events.log', 
        maxBytes=5*1024*1024, # 5 MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    file_handler.addFilter(EventFilter()) # Use the new, strict event filter

    # 2. Handler for the console (operational logs)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(console_formatter)
    stream_handler.addFilter(AppFilter()) # Use the app filter that excludes events

    # Add our filtered handlers to the root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    # Set higher logging levels for noisy third-party libraries
    # (Their logs will be evaluated by the root logger's handlers)
    logging.getLogger("streamlit").setLevel(logging.WARNING)
    logging.getLogger("watchdog").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("ultralytics").setLevel(logging.WARNING)

    logging.info("Logging configured. File log for critical events, console for operational status.") 