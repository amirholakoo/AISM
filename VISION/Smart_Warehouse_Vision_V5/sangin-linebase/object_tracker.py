"""
Object Tracker Module
Handles object matching and dual-line counting logic
"""

import math

print("object-tracker-start")


class ObjectTracker:
    """Centroid-based tracker with counting line and class-specific counters"""
    
    def __init__(self, primary_line_x, secondary_line_x, config):
        """
        Initialize object tracker
        
        Args:
            primary_line_x: X position of the primary counting line (pixels)
            secondary_line_x: X position of the secondary counting line (pixels)
            config: Configuration object
        """
        self.counting_line_x = secondary_line_x
        self.counting_zone_margin = config.COUNTING_ZONE_MARGIN
        self.match_distance = config.MATCH_DISTANCE
        self.max_missed_frames = config.MAX_MISSED_FRAMES

        self.forklift_class_name = getattr(config, "FORKLIFT_CLASS_NAME", "forklift").lower()
        
        self.tracks = {}
        self.next_track_id = 0

        self.entry_count = 0
        self.exit_count = 0

        self.loaded_count = 0
        self.unloaded_count = 0

        self.events = []
        self.live_status = 1
    
    def update(self, detections, frame_index):
        """
        Update tracker with new detections.

        Args:
            detections: List of detection dictionaries
            frame_index: Current frame number

        Returns:
            Dict with tracked objects ready for annotation, current counts,
            and per-frame crossing events.
        """
        # Reset per-frame events
        self.events = []

        for track in self.tracks.values():
            track["matched"] = False
        
        tracked_visuals = []
        
        for det in detections:
            cx, cy = det["cx"], det["cy"]
            best_id = self._find_best_track(cx, cy)
            
            if best_id is None:
                best_id = self._create_track(cx, cy, det["class_name"], frame_index)
            
            track = self.tracks[best_id]
            track["matched"] = True
            track["missed"] = 0
            track["last_seen"] = frame_index
            track["prev_x_prev"] = track["prev_x"]
            track["prev_y_prev"] = track["prev_y"]
            track["prev_x"] = cx
            track["prev_y"] = cy
            track["class_name"] = det["class_name"]
            track["bbox_points"] = det["bbox_points"]
            track["conf"] = det["conf"]
            self._check_crossings(best_id, track)
            
            tracked_visuals.append({
                "track_id": best_id,
                "cx": cx,
                "cy": cy,
                "bbox_points": det["bbox_points"],
                "class_name": det["class_name"],
                "conf": det["conf"],
                "counted": track["counted"],
            })
        
        self._purge_stale_tracks()

        return {
            "tracks": tracked_visuals,
            "counts": {
                "entry": self.entry_count,
                "exit": self.exit_count,
                "loaded": self.loaded_count,
                "unloaded": self.unloaded_count,
                "live": self.live_status,
            },
            "events": list(self.events),
        }
    
    def _find_best_track(self, cx, cy):
        best_id = None
        best_distance = self.match_distance
        
        for track_id, track in self.tracks.items():
            distance = math.hypot(cx - track["prev_x"], cy - track["prev_y"])
            if distance < best_distance:
                best_distance = distance
                best_id = track_id
        
        return best_id
    
    def _create_track(self, cx, cy, class_name, frame_index):
        self.next_track_id += 1
        self.tracks[self.next_track_id] = {
            "prev_x": cx,
            "prev_y": cy,
            "prev_x_prev": cx,
            "prev_y_prev": cy,
            "class_name": class_name,
            "counted": False,
            "bbox_points": None,
            "matched": True,
            "missed": 0,
            "last_seen": frame_index,
            "conf": 0.0,
        }
        return self.next_track_id
    
    def _check_crossings(self, track_id, track):
        if track["counted"]:
            return
            
        prev_prev_x = track["prev_x_prev"]
        prev_x = track["prev_x"]
        
        if prev_prev_x is None:
            return

        line_rtl = prev_prev_x > self.counting_line_x and prev_x <= self.counting_line_x
        line_ltr = prev_prev_x < self.counting_line_x and prev_x >= self.counting_line_x
        
        if line_ltr and abs(prev_x - self.counting_line_x) <= self.counting_zone_margin:
            track["counted"] = True
            self.entry_count += 1
            self.unloaded_count += 1
            if str(track["class_name"]).lower() == self.forklift_class_name:
                self.live_status = 1
            self.events.append({
                "track_id": track_id,
                "frame_number": track.get("last_seen"),
                "class_name": track["class_name"],
                "direction": "left_to_right",
                "event_type": "entry",
            })

        elif line_rtl and abs(prev_x - self.counting_line_x) <= self.counting_zone_margin:
            track["counted"] = True
            self.exit_count += 1
            self.loaded_count += 1
            if str(track["class_name"]).lower() == self.forklift_class_name:
                self.live_status = 1
            self.events.append({
                "track_id": track_id,
                "frame_number": track.get("last_seen"),
                "class_name": track["class_name"],
                "direction": "right_to_left",
                "event_type": "exit",
            })
    
    def _purge_stale_tracks(self):
        stale_ids = [
            track_id
            for track_id, track in self.tracks.items()
            if not track["matched"]
        ]
        
        for track_id in stale_ids:
            track = self.tracks[track_id]
            track["missed"] += 1
            if track["missed"] > self.max_missed_frames:
                del self.tracks[track_id]

