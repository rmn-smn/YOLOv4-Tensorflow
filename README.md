# YOLOv4-Tensorflow

Yet another implementation of YOLOv4 in Tensorflow 2.x. Implemented mainly for self-educational purposes, using Keras's subclassing API.

## Run image detect using original Darknet weights

- download original weights from [AlexeyAB](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights)
- run image_detect_coco.py

## Retrain using coco data set

- download [COCO](https://cocodataset.org/#home) dataset
- place training and test data in model_data folder
- run run_training.py

## Train custom data

- refer to .names and .txt files in model_data/coco to prepare classes and annotations for custom data.

## Use the GUI to train or use the network

- run app.py
- Train tab: for training the model
  - select training and test data (.txt files)
  - select classes (.name file)
  - (optional) select original darknet weights or a TF checkpoint
  - tune training settings or use defaults
  - run training
- Test tab: for object detection with trained network
  - select image(s)
  - select classes (.name file)
  - load original darknet weights or a TF checkpoint
  - run detection

## Some references

Model:

[AlexeyAB](https://github.com/AlexeyAB/darknet)

[TensorFlow-2.x-YOLOv3](https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3)

Frontend:

[neural-network-sandbox](https://github.com/seanwu1105/neural-network-sandbox)

[Stackoverflow](https://stackoverflow.com/questions/48425316/how-to-create-pyqt-properties-dynamically/48432653#48432653)

