"""
Object Detector Module
Handles YOLO-based oriented bounding-box detection
"""

from pathlib import Path

import numpy as np
from ultralytics import YOLO

print("object-detector-start")

try:
    import torch
except ImportError:
    torch = None


class ObjectDetector:
    """YOLO-based object detector tailored for bale counting"""

    def __init__(self, config):
        """
        Initialize object detector

        Args:
            config: Configuration object
        """
        self.imgsz = config.DETECTION_IMAGE_SIZE
        self.conf_threshold = config.CONFIDENCE_THRESHOLD
        self.max_detections = getattr(config, "MAX_DETECTIONS", 30)
        self.exclude_classes = [cls.lower() for cls in config.EXCLUDE_CLASSES]
        self.model_path = Path(config.MODEL_PATH)
        self.uses_ncnn = self._is_ncnn_model(self.model_path)
        self.model = YOLO(str(self.model_path))
        self.device = None
        self.use_torch_context = torch is not None and not self.uses_ncnn
        if self.use_torch_context:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
    
    def detect(self, frame):
        """
        Run YOLO detection on a frame

        Args:
            frame: Input frame (numpy array)

        Returns:
            List of detection dictionaries with cx, cy, bbox points, class name, confidence
        """
        predict_kwargs = {
            "verbose": False,
            "imgsz": self.imgsz,
            "conf": self.conf_threshold,
            "max_det": self.max_detections,
        }

        detections = []
        if self.use_torch_context:
            with torch.inference_mode():
                results = self.model(frame, **predict_kwargs)
        else:
            results = self.model(frame, **predict_kwargs)

        if not results:
            return detections

        result = results[0]
        if not hasattr(result, "boxes") or result.boxes is None or len(result.boxes) == 0:
            return detections

        # Use standard YOLO boxes
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()

        for bbox, cls_id, conf in zip(boxes, classes, confidences):
            class_name = self.model.names[int(cls_id)].lower()
            if class_name in self.exclude_classes:
                continue

            x1, y1, x2, y2 = bbox
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Convert standard YOLO bbox values to rectangle points for drawing
            bbox_points = np.array(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                dtype=np.int32,
            )

            detections.append({
                "cx": cx,
                "cy": cy,
                "bbox_points": bbox_points,
                "class_name": class_name,
                "conf": float(conf),
            })

        return detections
    
    def _is_ncnn_model(self, path: Path):
        if path.is_dir():
            return any(path.glob("*.ncnn.param"))
        suffix = path.suffix.lower()
        return suffix in {".ncnn", ".param", ".bin"}
    
    def _to_numpy(self, value):
        data = value
        if hasattr(data, "cpu"):
            data = data.cpu()
        if hasattr(data, "numpy"):
            return data.numpy()
        return np.asarray(data)

