#!/bin/bash

# Hardcoded image name
IMAGE_NAME="svarogcldo/pytorch_2_0:1.0"

# Default to use port forwarding
USE_PORT_FORWARDING=0

while getopts ":n:p:" opt; do
  case $opt in
    n) CONTAINER_NAME="$OPTARG"
    ;;
    p) USE_PORT_FORWARDING="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

# Check if a container name is provided
if [ -z "$CONTAINER_NAME" ]; then
    echo "You must provide a container name using the -n option."
    echo "Usage: $0 -n <container_name> [-p 0/1 (1 to enable port forwarding, 0 to disable)]"
    exit 1
fi

PORT_FORWARDING=""
if [ "$USE_PORT_FORWARDING" -eq 1 ]; then
    PORT_FORWARDING="-p 8888:8888 -p 6006:6006"
fi

# Docker run command
docker run --rm -it --ipc=host --gpus all $PORT_FORWARDING -v $PWD:/workspace -v $PWD/datasets:/datasets --name $CONTAINER_NAME $IMAGE_NAME
