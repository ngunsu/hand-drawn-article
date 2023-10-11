from ultralytics.utils.benchmarks import benchmark

benchmark(model='./yolo5mu/1/weights/best.pt', data='./data/f4_1.yaml', imgsz=640, half=False, device=0)
