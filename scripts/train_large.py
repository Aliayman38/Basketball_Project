from ultralytics import YOLO

# Load the Large model architecture
model = YOLO("yolo11l.pt")

# Execute training with adjusted parameters for heavy workloads
model.train(
    data="data/dataset/data.yaml",
    epochs=25,
    imgsz=640,
    device=0,
    batch=8,
    workers=4,
    project="runs/train",
    name="basketball_large"
)