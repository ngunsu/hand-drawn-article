from ultralytics import RTDETR

model = RTDETR("./rtdetr-l/1/weights/last.pt")
model.info()
model.val(data="./data/f4_1.yaml", conf=0.25, imgsz=640, iou=0.6, split='test')

model = RTDETR("./rtdetr-l/2/weights/last.pt")
model.info()
model.val(data="./data/f4_1.yaml", conf=0.25, imgsz=640, iou=0.6, split='test')

model = RTDETR("./rtdetr-l/3/weights/last.pt")
model.info()
model.val(data="./data/f4_1.yaml", conf=0.25, imgsz=640, iou=0.6, split='test')

model = RTDETR("./rtdetr-l/4/weights/last.pt")
model.info()
model.val(data="./data/f4_1.yaml", conf=0.25, imgsz=640, iou=0.6, split='test')

model = RTDETR("./rtdetr-l/5/weights/last.pt")
model.info()
model.val(data="./data/f4_1.yaml", conf=0.25, imgsz=640, iou=0.6, split='test')
