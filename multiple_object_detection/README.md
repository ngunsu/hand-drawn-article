# Multiple Object Detection

## Introduction

Repository for symbol object detection: training and evaluation tools

## Requirements

The dataset must be inside a folder named datasets in the current directory

## Usage

- Launch the docker instance

```bash
./docker/launch.sh
```

- Go to project workspace (inside docker)

```bash
cd /fondef_workspace
```

## Training

Use the scripts to train the different models

- Example of usage

```bash
./scripts/train/train_yolo8m.sh
```

## Evaluation

Use the scripts to eval the different models

- Example of usage

```bash
./scripts/eval/eval_yolo8m.sh
```

Use python for the RT-DETR

## Generate images

- Use the scripts to predict and generate test processed data of the different models

```bash
python ./scripts/predict/predict_yolo8m.py
```

The prediction is generated in /usr/src/ultralytics/runs/detect/predict/
 **Remember to erase, after cp**

## Benchmark speed (easy) and convert models

- Use python scripts inside benchmarks

```bash
python ./scripts/benchmark/yolo8m.py
```

## Benchmark using trtexec

First, launch docker instance

```bash
docker run -it --rm --gpus all -v $PWD:/workspace nvcr.io/nvidia/tensorrt:23.07-py3
```

Then, go to trtexec folder

```bash
cd /usr/src/tensorrt/bin
```

Finally run trtexec, for example

```bash
./trtexec --onnx=/workspace/weights/yolo5nu.onnx --saveEngine=/tmp/yolo5nu.engine --avgRuns=1000
```
