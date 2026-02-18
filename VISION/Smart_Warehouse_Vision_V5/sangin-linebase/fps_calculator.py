"""
FPS Calculator Module
Handles FPS calculation and smoothing
"""

import time
from collections import deque

print("fps start")

class FPSCalculator:
    """FPS calculator with smoothing"""
    
    def __init__(self, smoothing_frames=30):
        self.smoothing_frames = smoothing_frames
        self.timestamps = deque(maxlen=smoothing_frames)
        
    def start_frame(self):
        return
    
    def end_frame(self):
        now = time.time()
        self.timestamps.append(now)
        
        if len(self.timestamps) >= 2:
            duration = self.timestamps[-1] - self.timestamps[0]
            if duration > 0:
                return (len(self.timestamps) - 1) / duration
        
        return 0.0
    
    def get_current_fps(self):
        if len(self.timestamps) >= 2:
            duration = self.timestamps[-1] - self.timestamps[0]
            if duration > 0:
                return (len(self.timestamps) - 1) / duration
        return 0.0

