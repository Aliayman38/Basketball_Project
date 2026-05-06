import cv2
import os
from detection.detector import BasketballDetector

def main():
    # المسارات بناءً على هيكلية مجلداتك
    video_path = 'data/video_3.mp4' 
    model_path = 'models/weights/last.pt'
    output_path = 'runs/detection.mp4'
    
    os.makedirs('runs', exist_ok=True)

    detector = BasketballDetector(model_path)
    cap = cv2.VideoCapture(video_path)
    
    # الحصول على خصائص الفيديو
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    print(f"🚀 Starting detection using classes: {detector.model.names}")

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # تشغيل الكشف
        detections = detector.detect_frame(frame, conf=0.3)

        # رسم المربعات يدوياً كما في الكود الناجح[cite: 2]
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            color = det["color"]
            label = f"{det['class_name']} {det['conf']:.2f}"

            # رسم المربع[cite: 2]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # رسم خلفية النص[cite: 2]
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            
            # كتابة النص[cite: 2]
            cv2.putText(frame, label, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        writer.write(frame)
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"Frame {frame_count} processed...")

    cap.release()
    writer.release()
    print(f"✅ Success! Result saved at {output_path}")

if __name__ == "__main__":
    main()