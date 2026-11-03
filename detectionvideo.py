from ultralytics import YOLO

# Load trained model
model = YOLO("best.pt")

# Run video detection
results = model.predict(
    source=r"C:\Users\hp\OneDrive\Desktop\Road-Hazard-Detection\prediction\flc.mp4",
    imgsz=640,
    conf=0.25,
    save=True,
    device="cpu"
)

print("Video detection completed!")