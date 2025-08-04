import cv2
import numpy as np
import time
from datetime import datetime
from pyzbar.pyzbar import decode
from config.settings import QRScannerConfig

class QRScanner:
    """
    A fast QR code scanner that relies exclusively on the pyzbar library for high-speed detection.
    This class is responsible for processing frames and identifying QR codes.
    """
    def __init__(self, config: QRScannerConfig):
        self.config = config
        self.console_cache = {}
        # This set tracks codes found within the current run to avoid re-processing.
        self.found_codes_this_run = set()

    def _is_in_console_cache(self, content):
        """Check if a QR code is still within its timeout period for console logging."""
        return content in self.console_cache and (time.time() - self.console_cache[content]) < self.config.CONSOLE_LOG_TIMEOUT

    def _update_console_cache(self, content):
        """Update the console cache with the latest timestamp."""
        self.console_cache[content] = time.time()

    def _preprocess_image(self, frame):
        """
        Apply a series of preprocessing filters to the image to enhance QR code visibility.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        yield gray
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yield clahe.apply(gray)
        yield cv2.GaussianBlur(gray, (5, 5), 0)

    def _draw_detection(self, frame, points, content, color=(0, 255, 0)):
        """Draw the bounding box and text for a detected QR code."""
        points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [points], isClosed=True, color=color, thickness=2)
        text_origin = (points[0][0][0], points[0][0][1] - 10)
        cv2.putText(frame, content, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def process_frame(self, frame):
        """
        Processes a single frame to find a QR code.
        Returns detection info if a NEW code is found, otherwise returns the processed frame.
        """
        display_frame = frame.copy()
        found_qr = None
        
        for image in self._preprocess_image(frame):
            decoded_objects = decode(image)
            if decoded_objects:
                main_object = decoded_objects[0]
                content = main_object.data.decode('utf-8')
                points = [point for point in main_object.polygon]
                found_qr = {"points": points, "content": content}
                break
        
        if found_qr:
            content = found_qr["content"]
            self._draw_detection(display_frame, found_qr["points"], content)
            
            if not self._is_in_console_cache(content):
                self._update_console_cache(content)
                print(f"✅ QR Code Detected: {content}")

            # If this is the first time seeing this code in this run, return its data.
            if content not in self.found_codes_this_run:
                self.found_codes_this_run.add(content)
                return display_frame, {"content": content, "timestamp": datetime.now(self.config.TIMEZONE).isoformat()}
        
        return display_frame, None 