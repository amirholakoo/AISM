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
        self.found_codes_this_run = set()

        self.qr_candidate = None
        self.candidate_frame_counter = 0
        self.candidate_last_center = None
        self.last_processed_content = None

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

    def _draw_detection(self, frame, points, content, color=(0, 255, 0), distance_cm=None):
        """Draw the bounding box and text for a detected QR code, including distance."""
        points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [points], isClosed=True, color=color, thickness=2)
        
        display_text = content
        if distance_cm is not None:
            display_text = f"{content} ({distance_cm:.1f} cm)"

        text_origin = (points[0][0][0], points[0][0][1] - 10)
        cv2.putText(frame, display_text, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def _add_distance_overlay(self, frame, distance_cm, largest_qr):
        """Add distance information overlay to the frame, positioned to avoid FPS text."""
        h, w = frame.shape[:2]
        

        overlay_width = 350
        overlay_height = 70
        start_x = w - overlay_width - 10  # 10px margin from right edge
        start_y = 10  # 10px margin from top
       
        overlay = np.zeros((overlay_height, overlay_width, 3), dtype=np.uint8)
        overlay[:] = (40, 40, 40)  
        
        # Distance text
        if distance_cm is not None:
            distance_text = f"Distance: {distance_cm:.1f} cm"
            distance_color = (0, 255, 0) if distance_cm <= self.config.MAX_SCAN_DISTANCE_CM else (0, 0, 255)
            

            if largest_qr:
                if distance_cm <= self.config.MAX_SCAN_DISTANCE_CM:
                    if self.candidate_frame_counter >= self.config.VALIDATION_FRAME_COUNT:
                        status_text = "READY TO SCAN"
                        status_color = (0, 255, 0)
                    else:
                        status_text = f"VALIDATING {self.candidate_frame_counter}/{self.config.VALIDATION_FRAME_COUNT}"
                        status_color = (0, 255, 255)
                else:
                    status_text = "TOO FAR"
                    status_color = (0, 0, 255)
            else:
                status_text = "NO QR CODE"
                status_color = (128, 128, 128)
        else:
            distance_text = "Distance: N/A"
            distance_color = (128, 128, 128)
            status_text = "NO QR CODE"
            status_color = (128, 128, 128)
        

        cv2.putText(overlay, distance_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, distance_color, 2)
        cv2.putText(overlay, status_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

        max_dist_text = f"Max: {self.config.MAX_SCAN_DISTANCE_CM}cm"
        cv2.putText(overlay, max_dist_text, (overlay_width - 80, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        

        end_x = min(start_x + overlay_width, w)
        end_y = min(start_y + overlay_height, h)
        actual_width = end_x - start_x
        actual_height = end_y - start_y
        
        if actual_width > 0 and actual_height > 0:

            alpha = 0.85
            frame[start_y:end_y, start_x:end_x] = cv2.addWeighted(
                frame[start_y:end_y, start_x:end_x], 1 - alpha, 
                overlay[:actual_height, :actual_width], alpha, 0
            )

    def process_frame(self, frame):
        """
        Processes a frame to find QR codes, validating based on size, stability, and calculated distance.
        """
        display_frame = frame.copy()

        all_decoded_objects = []
        for image in self._preprocess_image(frame):
            decoded_objects = decode(image)
            if decoded_objects:
                all_decoded_objects = decoded_objects
                break

        largest_qr = None
        max_area = 0
        
        for qr in all_decoded_objects:
            points = np.array([p for p in qr.polygon], dtype=np.int32)
            area = cv2.contourArea(points)
            
            if area > max_area:
                max_area = area
                largest_qr = qr

        validated_qr = None
        current_distance_cm = None
        if largest_qr:
            current_content = largest_qr.data.decode('utf-8')

            qr_points = np.array([p for p in largest_qr.polygon], dtype=np.int32)
            rect = cv2.minAreaRect(qr_points)
            apparent_width_px = max(rect[1])

            if apparent_width_px > 0:
                current_distance_cm = (self.config.KNOWN_QR_CODE_WIDTH_CM * self.config.FOCAL_LENGTH) / apparent_width_px
            
            is_close_enough = current_distance_cm is not None and current_distance_cm <= self.config.MAX_SCAN_DISTANCE_CM

            # Calculate the center of the current largest QR
            x_coords = [p.x for p in largest_qr.polygon]
            y_coords = [p.y for p in largest_qr.polygon]
            current_center = (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))

            is_same_candidate = self.qr_candidate and current_content == self.qr_candidate.data.decode('utf-8')
            is_stable = False
            if is_same_candidate and self.candidate_last_center:
                distance = np.sqrt((current_center[0] - self.candidate_last_center[0])**2 + (current_center[1] - self.candidate_last_center[1])**2)
                if distance < self.config.STABILITY_PIXEL_THRESHOLD:
                    is_stable = True

            if is_same_candidate and is_stable:
                self.candidate_frame_counter += 1
            else:
                self.qr_candidate = largest_qr
                self.candidate_frame_counter = 1
            
            self.candidate_last_center = current_center
            
            if is_close_enough and self.candidate_frame_counter >= self.config.VALIDATION_FRAME_COUNT:
                validated_qr = self.qr_candidate
        else:
            # If no QR is in view, reset everything
            self.qr_candidate = None
            self.candidate_frame_counter = 0
            self.candidate_last_center = None

        for qr in all_decoded_objects:
            content = qr.data.decode('utf-8')
            points = [point for point in qr.polygon]
            
            color = (0, 0, 255) # Red for non-candidates
            distance_cm = None

            qr_points_draw = np.array([p for p in qr.polygon], dtype=np.int32)
            rect_draw = cv2.minAreaRect(qr_points_draw)
            apparent_width_px_draw = max(rect_draw[1])
            if apparent_width_px_draw > 0:
                distance_cm = (self.config.KNOWN_QR_CODE_WIDTH_CM * self.config.FOCAL_LENGTH) / apparent_width_px_draw

            is_candidate = self.qr_candidate and content == self.qr_candidate.data.decode('utf-8')

            if is_candidate:
                is_close_enough_candidate = current_distance_cm is not None and current_distance_cm <= self.config.MAX_SCAN_DISTANCE_CM
                if not is_close_enough_candidate:
                    color = (255, 0, 0) 
                else:
                    color = (0, 255, 255) 

            if validated_qr and content == validated_qr.data.decode('utf-8'):
                color = (0, 255, 0) # Green for the validated QR code
            
            self._draw_detection(display_frame, points, content, color, distance_cm=distance_cm)

        self._add_distance_overlay(display_frame, current_distance_cm, largest_qr)

        if validated_qr:
            content = validated_qr.data.decode('utf-8')
            
            if content != self.last_processed_content:
                if not self._is_in_console_cache(content):
                    self._update_console_cache(content)
                    print(f"✅ QR Code Validated: {content}")

                self.last_processed_content = content
                self.found_codes_this_run.add(content)
                
                self.qr_candidate = None
                self.candidate_frame_counter = 0
                self.candidate_last_center = None
                return display_frame, {"content": content, "timestamp": datetime.now(self.config.TIMEZONE).isoformat()}
        
        return display_frame, None 