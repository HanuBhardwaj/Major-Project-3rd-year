from ultralytics import YOLO
# Loading the pretrained YOLO model
model = YOLO("yolo11n.pt")
# Detecting objects in an image
results = model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    show=True,
    conf=0.4
)
print("Detection completed!")