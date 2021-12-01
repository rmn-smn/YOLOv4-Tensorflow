# YOLOv4-Tensorflow

Yet another implementation of YOLOv4 in Tensorflow 2.x. Implemented mainly for self-educational purposes, using Keras's subclassing API.

## Run image detect using original Darknet weights

- download original weights from [AlexeyAB](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights)
- run image_detect_coco.py

## retrain using coco data set

- download [COCO](https://cocodataset.org/#home) dataset
- place training and test data in model_data folder
- run run_training.py

## train custom data

- refer to .names and .txt files in model_data/coco to prepare classes and annotations for custom data.
