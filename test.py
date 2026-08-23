import os
from ultralytics import YOLO
from PIL import Image  # Built-in library to open and display images

def test_inference():
    WEIGHTS_PATH = os.path.join("runs", "detect", "train-3", "weights", "best.pt")
    TEST_IMAGE_PATH = r"C:\Users\abhir\projects\car_crash\test\images\69_jpeg.rf.0b51f849de0ab0e1ef07a53fca1e1903.jpg" 

    if not os.path.exists(WEIGHTS_PATH):
        print(f"Error: Could not find best.pt at {WEIGHTS_PATH}")
        return
        
    model = YOLO(WEIGHTS_PATH)

    print(f"Running inference on: {TEST_IMAGE_PATH}")
    results = model.predict(
        source=TEST_IMAGE_PATH,
        conf=0.25,      
        save=True,      # Saves the image to runs/detect/predict-X
        device="0"      
    )

    # 5. Extract the saved directory path and display the image automatically
    saved_dir = results[0].save_dir
    saved_image_name = os.path.basename(TEST_IMAGE_PATH)
    full_output_path = os.path.join(saved_dir, saved_image_name)

    if os.path.exists(full_output_path):
        print(f"\nOpening detected image automatically: {full_output_path}")
        # This opens the image using your default Windows photo viewer app instantly
        img = Image.open(full_output_path)
        img.show()
    else:
        print(f"Could not locate the saved image at {full_output_path}")

if __name__ == "__main__":
    test_inference()