"""
Quick test — runs YOLO on your video and draws raw boxes directly.
No tracker, no clustering, no interpolation.
If you see boxes here, YOLO works fine.
"""
import cv2
import argparse
from ultralytics import YOLO

COLORS = {
    "basketball": (0, 200, 255),
    "net":        (255, 165,   0),
    "player":     (0, 255,   0),
    "referee":    (0, 255, 255),
}

def run(video_path, model_path, output_path, conf=0.3):
    model  = YOLO(model_path)
    cap    = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    import os; os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf, verbose=False)[0]

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf_  = float(box.conf[0])
            name   = model.names[cls_id]
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            color  = COLORS.get(name, (255,255,255))

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            label = f"{name} {conf_:.2f}"
            (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+6, y1), color, -1)
            cv2.putText(frame, label, (x1+3, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 50 == 0:
            n = len(results.boxes)
            print(f"Frame {frame_idx}  detections={n}")

    cap.release()
    writer.release()
    print(f"\nDone → {output_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",  required=True)
    ap.add_argument("--model",  required=True)
    ap.add_argument("--output", default="runs/output/test_det.mp4")
    ap.add_argument("--conf",   type=float, default=0.3)
    args = ap.parse_args()
    run(args.video, args.model, args.output, args.conf)
