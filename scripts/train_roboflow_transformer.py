from ultralytics import RTDETR # Roboflow uses Ultralytics backend for this in their tutorials

def main():
    print("🚀 Starting Roboflow Detection Transformer Training...")

    model = RTDETR('rtdetr-l.pt')

    results = model.train(
        data='data/dataset/data.yaml',
        epochs=50,
        imgsz=1280,
        batch=4,
        project='runs/train',
        name='roboflow_transformer_v1'
    )

if __name__ == '__main__':
    main()
