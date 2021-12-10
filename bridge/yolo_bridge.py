import time
import threading
import PyQt5.QtCore

from . import Bridge, BridgeProperty
from .observer import Observable
from model.yolo import YOLOv4
from model.trainer import Initializer, Trainer
from model.dataset import Dataset
from model.utils import detect_image


class YoloBridge(Bridge):
    '''
    Implments Bridge for communication between 
    Yolo Tensorflow Model and QT Frontend
    '''
    # Bridge properties
    ui_refresh_interval = BridgeProperty(0.0)
    classes = BridgeProperty('')
    train_annotations = BridgeProperty('')
    test_annotations = BridgeProperty('')
    epochs = BridgeProperty(100)
    warmup_epochs = BridgeProperty(2)
    learning_rate = BridgeProperty(1e-6)
    initial_lr = BridgeProperty(1e-4)
    logdir = BridgeProperty('log')
    checkpointdir = BridgeProperty('')
    save_checkpoints = BridgeProperty(True)
    save_best_checkpoint_only = BridgeProperty(True)

    current_iter = BridgeProperty(0)
    current_epoch = BridgeProperty(0)
    current_learning_rate = BridgeProperty(0.0)
    current_train_giou_loss = BridgeProperty(0.0)
    current_train_conf_loss = BridgeProperty(0.0)
    current_train_prob_loss = BridgeProperty(0.0)
    current_train_total_loss = BridgeProperty(0.0)
    current_test_giou_loss = BridgeProperty(0.0)
    current_test_conf_loss = BridgeProperty(0.0)
    current_test_prob_loss = BridgeProperty(0.0)
    current_test_total_loss = BridgeProperty(0.0)
    steps_per_epoch = BridgeProperty(0)

    batch_size = BridgeProperty(4)
    data_augmentation = BridgeProperty(True)

    load_darknet_weights = BridgeProperty(False)
    load_checkpoint = BridgeProperty(False)
    weights_path = BridgeProperty('')
    checkpoint_dir = BridgeProperty('')

    detect_image_path = BridgeProperty([])
    annotated_image_path = BridgeProperty([])

    status = BridgeProperty('detector not yet initialized')

    def __init__(self):
        super().__init__()
        self.model = None
        self.trainer = None
        self.initializer = None
        self.trainset = None
        self.testset = None

    @PyQt5.QtCore.pyqtSlot()
    def yolo_initialize(self):
        # call yolo model

        self.model = None
        print(self.classes)
        self.model = ObservableYolo(
            self,
            self.ui_refresh_interval,
            classes=self.classes
        )

    @PyQt5.QtCore.pyqtSlot()
    def yolo_load(self):
        # call yolo observable initializer and load weights/checkpoint
  
        self.initializer = ObservableInitializer(
            self,
            self.ui_refresh_interval,        
            model = self.model,        
            checkpointdir = self.checkpoint_dir,
            darknetweights = self.weights_path
            )

        if self.load_darknet_weights:
            self.status = 'loading darknet weights'
            self.initializer.load_darknet_weights()
        if self.load_checkpoint:
            self.status = 'loading checkpoint'
            self.initializer.load_checkpoint()

    @PyQt5.QtCore.pyqtSlot(int)
    def yolo_detect(self,iterator: int):
        # run detection

        self.annotated_image_path = \
            [i.split('.')[0]+'_annotated.'+i.split('.')[1] 
            for i in self.detect_image_path]
        
        detect_image(
            self.model, 
            image_path = str(self.detect_image_path[iterator]),
            output_path = str(self.annotated_image_path[iterator]),
            classes=self.classes, 
            input_size=416, 
            show=False, 
            rectangle_colors=(255,0,0)
            )

    @PyQt5.QtCore.pyqtSlot()
    def yolo_train(self):
        # load observable dataset, trainer and run training
        
        self.yolo_initialize()

        self.trainset = ObservableDataset(
            self,
            self.ui_refresh_interval,
            dataset_type = 'train', 
            annotations = self.train_annotations, 
            classes = self.classes,
            batch_size = self.batch_size,
            data_augmentation = self.data_augmentation
            )

        self.testset = ObservableDataset(
            self,
            self.ui_refresh_interval,
            dataset_type = 'test', 
            annotations = self.test_annotations, 
            classes = self.classes,
            batch_size = self.batch_size,
            data_augmentation = self.data_augmentation
            )

        if self.initializer is None \
            and (self.load_darknet_weights or self.load_checkpoint):
                self.yolo_load()

        self.trainer = ObservableTrainer(
            self,
            self.ui_refresh_interval,
            model = self.model, 
            trainset = self.trainset, 
            testset  = self.testset,
            epochs = self.epochs, 
            warmup_epochs = self.warmup_epochs, 
            learning_rate = self.learning_rate, 
            initial_lr = self.initial_lr,
            logdir = self.logdir, 
            checkpointdir = self.checkpointdir, 
            save_checkpoints = self.save_checkpoints,
            save_best_checkpoint_only = self.save_best_checkpoint_only
        )

        self.status = 'training ...'
        self.trainer.train()
        self.status = 'stopped training'
        time.sleep(1)
        self.status = 'ready to train'

        return
        
    @PyQt5.QtCore.pyqtSlot()
    def start_training(self):
        # call yolo_train in a new thread

        thread = threading.Thread(target=self.yolo_train)
        thread.start()

    @PyQt5.QtCore.pyqtSlot()
    def stop_training(self):
        # interrupt training

        try:
            self.trainer.interrupt = True
            self.status = 'interrupting training...'
        except AttributeError:
            self.status = 'detector not yet initialized'



class ObservableYolo(Observable, YOLOv4):
    # construct a observable version of the YOLOv4 class 
    # (currently not really necessary)

    def __init__(self, observer, ui_refresh_interval, **kwargs):
        Observable.__init__(self, observer)
        YOLOv4.__init__(self, **kwargs)
        self.ui_refresh_interval = ui_refresh_interval

    def __setattr__(self, name, value):
        super().__setattr__(name, value)

class ObservableTrainer(Observable, Trainer):
    # construct an observable version of the Trainer class

    def __init__(self, observer, ui_refresh_interval, **kwargs):
        Observable.__init__(self, observer)
        Trainer.__init__(self, **kwargs)
        self.ui_refresh_interval = ui_refresh_interval

    def __setattr__(self, name, value):
        # overload __setattr__ to fire a notification every time 
        # a variable corresponding to a respective bridge variable is changed.
        super().__setattr__(name, value)
        if name == 'current_iter':
            self.notify(name, value)
        elif name == 'current_train_stats':
            self.notify(name, value)
        elif name == 'current_epoch':
            self.notify(name, value)
        elif name == 'current_iter':
            self.notify(name, value)
        elif name == 'steps_per_epoch':
            self.notify(name, value)
        elif name == 'current_learning_rate':
            self.notify(name, value)
        elif name == 'current_train_giou_loss':
            self.notify(name, value)
        elif name == 'current_train_conf_loss':
            self.notify(name, value)
        elif name == 'current_train_prob_loss':
            self.notify(name, value)
        elif name == 'current_train_total_loss':
            self.notify(name, value)
        elif name == 'current_test_giou_loss':
            self.notify(name, value)
        elif name == 'current_test_conf_loss':
            self.notify(name, value)
        elif name == 'current_test_prob_loss':
            self.notify(name, value)
        elif name == 'current_test_total_loss':
            self.notify(name, value)

class ObservableInitializer(Observable, Initializer):
    # construct an observable version of the Initializer class 
    # (currently not really necessary)

    def __init__(self, observer, ui_refresh_interval, **kwargs):
        Observable.__init__(self, observer)
        Initializer.__init__(self, **kwargs)
        self.ui_refresh_interval = ui_refresh_interval

    def __setattr__(self, name, value):
        super().__setattr__(name, value)

class ObservableDataset(Observable, Dataset):
    # construct an observable version of the Dataset class 
    # (currently not really necessary)

    def __init__(self, observer, ui_refresh_interval, **kwargs):
        Observable.__init__(self, observer)
        Dataset.__init__(self, **kwargs)
        self.ui_refresh_interval = ui_refresh_interval

    def __setattr__(self, name, value):
        super().__setattr__(name, value)

