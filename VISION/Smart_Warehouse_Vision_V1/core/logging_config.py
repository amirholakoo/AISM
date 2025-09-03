import logging
from logging.handlers import RotatingFileHandler
import sys

def setup_logging():
    """
    Configures logging to output to both a rotating file and the console.
    """

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


    file_handler = RotatingFileHandler(
        'warehouse_events.log', 
        maxBytes=5*1024*1024, # 5 MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logging.info("Logging configured successfully.") 