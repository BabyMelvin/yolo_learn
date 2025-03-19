from ultralytics import YOLO

# Load a model
model = YOLO("runs\detect\train\weights\last.pt")  # load a partially trained model

# resume training
# checkpoints are saved at the end of every epoch by default,
results = model.train(resume=True)