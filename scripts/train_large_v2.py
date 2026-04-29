from ultralytics import YOLO

# continue from v1
model = YOLO("runs/detect/runs/train/large_v1_640/weights/best.pt")
#1280
model.train(
    data="data/dataset/data.yaml",
    epochs=50,
    imgsz=1280,
    device=0,
    batch=4,
    project="runs/detect/runs/train",
    name="basketball_large_v2",
    lr0=0.0005,
    patience=15
)