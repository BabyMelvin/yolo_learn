from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
# To enable multi-GPU training, specify the GPU device IDs you wish to use.
results = model.train(data="coco8.yaml", epochs=10, imgsz=640, device=[0, 1])