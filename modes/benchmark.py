from  ultralytics.utils.benchmarks import benchmark

benchmark(model = 'yolo11n.pt', data = 'coco8.yaml', imgsz = 640, half = False, device = 'cpu')