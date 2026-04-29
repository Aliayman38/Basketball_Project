from ultralytics import YOLO
import os

def main():
    # 1. Load the model
    model = YOLO("yolo11n.pt")

    # 2. Train the model
    model.train(
        data="data/dataset/data.yaml",
        epochs=25,
        imgsz=640,
        device="cpu",
        project="runs/train",
        name="basketball_yolo11"
    )

if __name__ == "__main__":
    main()