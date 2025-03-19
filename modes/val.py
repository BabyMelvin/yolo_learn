from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load an official model

# Customize validation settings
# validation_results = model.val(data="coco8.yaml", imgsz=640, batch=16, conf=0.25, iou=0.6, device="0")
'''
YOLO11 model validation provides several key metrics to assess model performance. These include:

mAP50 (mean Average Precision at IoU threshold 0.5)
mAP75 (mean Average Precision at IoU threshold 0.75)
mAP50-95 (mean Average Precision across multiple IoU thresholds from 0.5 to 0.95)
'''
# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
print(metrics.box.map)  # mAP50-95
print(metrics.box.map50)  # mAP50
print(metrics.box.map75)  # mAP75
print(metrics.box.maps)  # list of mAP50-95 for each category