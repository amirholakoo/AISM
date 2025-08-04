import cv2
import time
import os
import threading
from queue import Queue, Empty, Full

class VideoStreamer:
    """
    Handles connecting to and reading from a video source in a separate thread.
    Includes connection retries and verification to ensure the stream is live.
    """
    def __init__(self, source):
        self.source = source
        self.cap = None
        self.is_running = False
        self.frame_queue = Queue(maxsize=2)
        self.grab_thread = None
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

    def connect(self):
        """
        Attempts to connect to the video source with retries and verification.
        A connection is only considered successful if a frame can be grabbed.
        """
        print(f"Attempting to connect to video source: {self.source}")
        max_retries = 10
        retry_delay = 3

        for attempt in range(max_retries):
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            if self.cap and self.cap.isOpened():
                grab_success = self.cap.grab()
                if grab_success:
                    print("✅ Connection successful. Allowing stream to stabilize...")
                    time.sleep(2.0) # Add a delay to allow the stream to stabilize
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
                    self.is_running = True
                    self.grab_thread = threading.Thread(target=self._grab_loop, daemon=True)
                    self.grab_thread.start()
                    return True
                else:
                    print(f"⚠️ Connection attempt {attempt + 1}/{max_retries}: Connected but failed to grab frame.")
                    self.cap.release()
            else:
                print(f"⚠️ Connection attempt {attempt + 1}/{max_retries}: Failed to open source.")

            time.sleep(retry_delay)

        print(f"❌ Failed to connect to video source after {max_retries} attempts.")
        self.is_running = False
        return False

    def _grab_loop(self):
        """Continuously grabs frames from the source and places them in the queue."""
        while self.is_running:
            if not self.cap.isOpened():
                print("⚠️ Stream disconnected. Stopping grab thread.")
                break

            ret = self.cap.grab()
            if not ret:
                continue

            _, frame = self.cap.retrieve()
            
            try:
                if self.frame_queue.full():
                    self.frame_queue.get_nowait()
                self.frame_queue.put(frame, block=False)
            except Full:
                pass

    def read(self):
        """Reads the latest frame from the queue."""
        if not self.is_running:
            return False, None
        
        try:
            return True, self.frame_queue.get(timeout=2.0)
        except Empty:
            return False, None

    def stop(self):
        """Stops the grabbing thread and releases the video capture object."""
        self.is_running = False
        if self.grab_thread is not None:
            self.grab_thread.join(timeout=2)
        if self.cap:
            self.cap.release()
        print("Video stream stopped.") 