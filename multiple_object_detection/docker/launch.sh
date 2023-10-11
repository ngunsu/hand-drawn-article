# Set image name as a variable
t=ultralytics/ultralytics:latest

# Pull the latest ultralytics image from Docker Hub
docker pull $t

# Run the ultralytics image in a container with GPU support
docker run -it --ipc=host --gpus all -v $PWD:/fondef_workspace -v $PWD/datasets/fondef_id20I10262/workpieces_yolov5_symbols/:/opt/datasets/fondef_id20I10262/workpieces_yolov5_symbols/ $t 
