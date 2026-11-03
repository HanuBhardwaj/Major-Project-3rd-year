from ultralytics import YOLO

# Load trained model
model = YOLO("best.pt")

# Image detection
results = model.predict(
    source=r"C:\Users\hp\OneDrive\Desktop\Road-Hazard-Detection\prediction\ptwl.webp",
    imgsz=640,
    conf=0.25,
    save=True,
    device="cpu"
)

# Display detected classes
for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        print(
            model.names[class_id],
            round(confidence * 100, 2),
            "%"
        )

print("Detection completed!")