from ultralytics.utils.benchmarks import benchmark

benchmark(model='./yolo5m6u/1/weights/best.pt', data='./data/f4_1.yaml', imgsz=1280, half=False, device=0)
