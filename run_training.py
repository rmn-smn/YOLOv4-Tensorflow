from model.trainer import Trainer,Initializer
from model.yolo import YOLOv4
from model.dataset import Dataset

def main():
    ## download coco data from https://cocodataset.org/#home
    ## place training data in "model_data/coco/train2017.txt"
    ## place training data in "model_data/coco/val2017.txt"

    classes = "model_data/coco/coco.names"
    train_annotations = "model_data/coco/train2017.txt"
    test_annotations = "model_data/coco/val2017.txt"

    yolo = YOLOv4(classes=classes)
    #initializer = Initializer(yolo)
    #initializer.load_darknet_weights()
    #initializer.load_checkpoint()
    
    trainset = Dataset(
        'train', annotations = train_annotations, 
        classes = classes,load_images_to_ram = False)
    testset = Dataset(
        'test', annotations = test_annotations, 
        classes = classes, load_images_to_ram = False)
    trainer = Trainer(yolo, trainset = trainset, testset  = testset)
    trainer.train()


if __name__ == '__main__':
    main()