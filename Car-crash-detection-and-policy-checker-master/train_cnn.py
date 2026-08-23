from ultralytics import YOLO

def main():
    # 1. Initialize the base pretrained YOLO model
    model = YOLO("yolov8n.pt") 

    # 2. Run training locally on your NVIDIA GeForce RTX 4050
    results = model.train(
        data="data.yaml",   # Path to your unzipped data.yaml file
        epochs=50,          # 50 epochs
        imgsz=640,          # Standard training resolution
        device="0",         # Uses your primary GPU
        batch=-1            # Auto-tunes the batch size to safely max out your VRAM
    )

if __name__ == '__main__':
    main()