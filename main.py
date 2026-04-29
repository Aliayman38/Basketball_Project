from src.detection.detector import BasketballDetector
import os

def main():
    model_path = "models/weights/best.pt"
    
    video_path = "data/test_video.mp4"
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print("Loading model and starting detection...")
    detector = BasketballDetector(model_path)
    detector.predict_video(source_path=video_path, save_dir="runs/detect")

if __name__ == "__main__":
    main()