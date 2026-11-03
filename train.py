from pathlib import Path
from ultralytics import YOLO

# Dataset configuration file
data_path = Path("datasets/pothole.yolov8/data.yaml").resolve()

if not data_path.exists():
    raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

# Load lightweight YOLO11n model
model = YOLO("yolo11n.pt")

# Train the model
results = model.train(
    data=str(data_path),
    epochs=30,
    imgsz=640,
    batch=4,
    workers=2,
    device="cpu",
    project="runs/pothole_detection",
    name="yolo11n_potholes"
)

print("Training completed!")
print("Best model: runs/pothole_detection/yolo11n_potholes/weights/best.pt")