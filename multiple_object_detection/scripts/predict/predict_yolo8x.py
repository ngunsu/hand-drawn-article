from ultralytics import YOLO

# Load a model
model = YOLO('./yolo8x/1/weights/last.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images
source = '/opt/datasets/fondef_id20I10262/workpieces_yolov5_symbols/test/images/'
model.predict(source, save=True, imgsz=640, conf=0.25, iou=0.6, half=True)
