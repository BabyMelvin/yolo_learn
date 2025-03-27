from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

results = model("https://ultralytics.com/images/bus.jpg")

# View the results
for result in results:
    print(result)

for result in results:
    print(result.boxes)