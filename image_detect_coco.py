import tensorflow as tf
from model.trainer import Initializer
from model.yolo import YOLOv4
from model.utils import detect_image

## download YOLOv4 Darknet weights (yolov4.weights) from 
# https://github.com/AlexeyAB/darknet and place in "model_data" folder

image_path   = "images/city.jpg"
coco_classes = "model_data/coco/coco.names"
darknetweights = "model_data/yolov4.weights"

model = YOLOv4(classes=coco_classes)
initializer = Initializer(model,darknetweights = darknetweights)
initializer.load_darknet_weights()

detect_image(model, image_path, "", classes=coco_classes, input_size=416, 
             show=True, rectangle_colors=(255,0,0))
