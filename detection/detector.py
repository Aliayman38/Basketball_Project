from ultralytics import YOLO, RTDETR
import cv2
import os

def _load_model(model_path: str):
    """Load YOLO or RT-DETR based on the weight filename."""
    name = os.path.basename(model_path).lower()
    if "rtdetr" in name or "rt-detr" in name or "rt_detr" in name:
        return RTDETR(model_path)
    return YOLO(model_path)

class BasketballDetector:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Weights file not found: {model_path}")
        
        self.model = _load_model(model_path)
        # الألوان التي نجحت في الاختبار
        self.colors = {
            "basketball": (0, 200, 255),
            "net":        (255, 165,   0),
            "player":     (0, 255,   0),
            "referee":    (0, 255, 255),
        }

    def detect_frame(self, frame, conf=0.3):
        """
        تشغيل الكشف على فريم واحد واسترجاع النتائج بتنسيق بسيط.
        """
        results = self.model(frame, conf=conf, verbose=False)[0]
        detections = []
        
        for box in results.boxes:
            cls_id = int(box.cls[0])
            name = self.model.names[cls_id]
            detections.append({
                "bbox": map(int, box.xyxy[0]),
                "conf": float(box.conf[0]),
                "class_id": cls_id,
                "class_name": name,
                "color": self.colors.get(name, (255, 255, 255))
            })
        return detections