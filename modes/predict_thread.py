from ultralytics import YOLO
from threading import Thread

def thread_safe_predict(module, img):
    model = YOLO(module)
    results = model.predict(img)
    # process results

# Starting threads that each have their own instance of the model
Thread(target=thread_safe_predict, args=("yolo11n.pt", img1)).start()
Thread(target=thread_safe_predict, args=("yolo11n.pt", img2)).start()