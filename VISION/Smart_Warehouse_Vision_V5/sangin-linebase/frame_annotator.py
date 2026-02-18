"""
Frame Annotator Module
Handles drawing annotations on frames
"""

import cv2

print("frame-annotation start")


class FrameAnnotator:
    """Frame annotation and visualization"""
    
    def __init__(self, config, primary_line_x, secondary_line_x):
        """
        Initialize frame annotator
        
        Args:
            config: Configuration object
            primary_line_x: Primary counting line position (pixels)
            secondary_line_x: Secondary counting line position (pixels)
        """
        self.config = config
        self.primary_line_x = primary_line_x
        self.secondary_line_x = secondary_line_x
        self.margin = config.COUNTING_ZONE_MARGIN
    
    def annotate_frame(self, frame, tracked_objects, counts, fps):
        """
        Annotate frame with detections, lines, and info
        
        Args:
            frame: Input frame
            tracked_objects: List of tracked object dicts
            counts: Dictionary with primary, secondary, total counts
            fps: Current FPS value
        """
        annotated_frame = frame.copy()
        
        self._draw_tracks(annotated_frame, tracked_objects)
        self._draw_counting_lines(annotated_frame)
        self._draw_info_overlay(annotated_frame, counts, fps)
        
        return annotated_frame
    
    def _draw_tracks(self, frame, tracked_objects):
        for track in tracked_objects:
            bbox_points = track["bbox_points"]
            color = self.config.COLOR_BBOX
            if track["counted"]:
                color = self.config.COLOR_COUNTED_OBJECT
            
            cv2.polylines(frame, [bbox_points], True, color, 2)
            cv2.circle(frame, (track["cx"], track["cy"]), 3, color, -1)
            label = f'{track["class_name"]} {track["conf"]:.2f}'
            text_origin = (track["cx"] + 5, track["cy"] - 5)
            cv2.putText(
                frame,
                label,
                text_origin,
                self.config.FONT,
                self.config.FONT_SCALE_SMALL,
                color,
                self.config.FONT_THICKNESS,
            )
    
    def _draw_counting_lines(self, frame):
        height = frame.shape[0]
        
        cv2.line(
            frame,
            (self.secondary_line_x, 0),
            (self.secondary_line_x, height),
            (0, 255, 255),
            3,
        )
    
    def _draw_info_overlay(self, frame, counts, fps):
        info_bg_color = (40, 40, 40)
        cv2.rectangle(frame, (5, 5), (300, 200), info_bg_color, -1)
        
        entry = counts.get("entry", 0)
        exit_count = counts.get("exit", 0)
        loaded = counts.get("loaded", 0)
        unloaded = counts.get("unloaded", 0)
        live = counts.get("live", 0)
        
        cv2.putText(
            frame,
            f'Entries: {entry}',
            (10, 30),
            self.config.FONT,
            0.7,
            (50, 255, 50),
            2,
        )
        cv2.putText(
            frame,
            f'Exits: {exit_count}',
            (10, 60),
            self.config.FONT,
            0.7,
            (50, 50, 255),
            2,
        )
        cv2.putText(
            frame,
            f'FPS: {fps:.1f}',
            (10, 90),
            self.config.FONT,
            0.7,
            (50, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f'Loaded: {loaded}',
            (10, 120),
            self.config.FONT,
            0.7,
            (50, 50, 255),
            2,
        )
        cv2.putText(
            frame,
            f'Unloaded: {unloaded}',
            (10, 150),
            self.config.FONT,
            0.7,
            (50, 255, 50),
            2,
        )
        cv2.putText(
            frame,
            f'Live: {live}',
            (10, 180),
            self.config.FONT,
            0.7,
            (0, 200, 200),
            2,
        )

