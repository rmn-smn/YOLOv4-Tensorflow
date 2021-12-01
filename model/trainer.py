
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import shutil
import numpy as np
import tensorflow as tf
from model.yolo import YOLOv4,YoloLoss
from model.utils import load_yolo_weights

class Initializer():
    '''
    Trainer for initializer for yolo model.

    Args: 
          model (Model):             Yolo model instance
          input_tensor (tf.tensor):  tensor defining the input dimensions
          checkpointdir (str):       path to checkpoint directory
          darknetweights (str):      path to yolo4.weights
    '''
    def __init__(self,model,input_tensor = tf.zeros((1,416,416,3)), 
                 checkpointdir = 'checkpoints'):

        self.checkpointdir = checkpointdir
        self.model = model
        self.input_tensor = input_tensor
        _ = self.model(self.input_tensor,training=True)

    def load_darknet_weights(self,darknetweights = "model_data/yolov4.weights"):

        yolo_coco_classes = 'model_data/coco/coco.names'
        Darknet = YOLOv4(classes=yolo_coco_classes)
        _=Darknet(self.input_tensor,training=True)
        print('loading Darknet weights: ' + darknetweights)
        load_yolo_weights(Darknet, darknetweights) 

        for i, l in enumerate(Darknet.layers):
            layer_weights = l.get_weights()
            if layer_weights != []:
                try:
                    self.model.layers[i].set_weights(layer_weights)
                except:
                    print("skipping", self.model.layers[i].name)

    def load_checkpoint(self):

        try:
            self.model.load_weights(f"{self.checkpointdir}/yolov4_custom")
            print(f"Loading checkpoint: {self.checkpointdir}/yolov4_custom")
        except (ValueError):
            print("Shapes are incompatible, training from scratch")



class Trainer():
    '''
    Trainer for yolo model.
    code adapted from: https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3

    Args: 
          model (Model):             Yolo model instance
          epochs (int):              number of training epochs
          warmup_epochs (int):       number of warmup epochs
          learning_rate (float):     final learning rate
          initial_lr (float):        initial learning rate
          logdir (str):              path to logfile directory
          checkpointdir (str):       path to checkpoint directory
          darknetweights (str):      path to yolo4.weights
          transfer_darknet_weights:  if true, original darknet weights 
                                     are loaded
          load_checkpoint:           if true, checkpoint is loaded
          save_checkpoints:          if true a checkpoint is saved after every 
                                     validation
          save_best_checkpoint_only: if true, only the checkpoint with the 
                                     lowest validatin loss is saved
    '''

    def __init__(self, model, trainset, testset, epochs = 100, warmup_epochs = 2, 
                 learning_rate = 1e-6, initial_lr = 1e-4,
                 logdir = 'log', checkpointdir = 'checkpoints', 
                 save_checkpoints = True,
                 save_best_checkpoint_only = True):

        gpus = tf.config.experimental.list_physical_devices('GPU')
        print(f'GPUs {gpus}')
        if len(gpus) > 0:
            try: tf.config.experimental.set_memory_growth(gpus[0], True)
            except RuntimeError: pass

        self.checkpointdir = checkpointdir
        self.logdir = logdir
        self.train_epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.save_checkpoints = save_checkpoints
        self.save_best_checkpoint_only = save_best_checkpoint_only
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.initial_lr = initial_lr
        
        if os.path.exists(logdir): shutil.rmtree(logdir)
        self.writer = tf.summary.create_file_writer(logdir)
        self.validate_writer = tf.summary.create_file_writer(logdir)

        self.trainset = trainset
        self.testset = testset

        self.steps_per_epoch = len(self.trainset)
        self.global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
        self.warmup_steps = self.warmup_epochs * self.steps_per_epoch
        self.total_steps = self.train_epochs * self.steps_per_epoch

        self.model = model

        # build model
        self.model.build(
            input_shape=(None,self.trainset.input_sizes,
            self.trainset.input_sizes,3))

        self.optimizer = tf.keras.optimizers.Adam()


    def train_step(self,image_data, target, loss_fn):
        with tf.GradientTape() as tape:
            pred_result = self.model(image_data, training=True)
            giou_loss=conf_loss=prob_loss=0

            # optimizing process
            grid = 3
            for i in range(grid):
                conv, pred = pred_result[i*2], pred_result[i*2+1]
                loss_items = loss_fn(pred, conv, *target[i], i)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss

            gradients = tape.gradient(
                total_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables))

            # update learning rate
            # about warmup: 
            # https://arxiv.org/pdf/1812.01187.pdf&usg=ALkJrhglKOPDjNt6SHGbphTHyMcT0cuMJg
            self.global_steps.assign_add(1)
            if self.global_steps < self.warmup_steps:
                lr = self.global_steps / self.warmup_steps * self.initial_lr
            else:
                lr = self.learning_rate + 0.5 * \
                    (self.initial_lr - self.learning_rate)*(
                    (1 + tf.cos((self.global_steps - self.warmup_steps) / 
                    (self.total_steps - self.warmup_steps) * np.pi)))
            self.optimizer.lr.assign(lr.numpy())

            # writing summary data
            with self.writer.as_default():
                tf.summary.scalar(
                    "lr", self.optimizer.lr, step=self.global_steps)
                tf.summary.scalar(
                    "loss/total_loss", total_loss, step=self.global_steps)
                tf.summary.scalar(
                    "loss/giou_loss", giou_loss, step=self.global_steps)
                tf.summary.scalar(
                    "loss/conf_loss", conf_loss, step=self.global_steps)
                tf.summary.scalar(
                    "loss/prob_loss", prob_loss, step=self.global_steps)
            self.writer.flush()
            
        return (self.global_steps.numpy(), self.optimizer.lr.numpy(), 
                giou_loss.numpy(), conf_loss.numpy(), prob_loss.numpy(), 
                total_loss.numpy())

    def validate_step(self,image_data, target, loss_fn):
        with tf.GradientTape() as tape:
            pred_result = self.model(image_data, training=False)

            giou_loss=conf_loss=prob_loss=0

            # optimizing process
            grid = 3
            for i in range(grid):
                conv, pred = pred_result[i*2], pred_result[i*2+1]
                loss_items = loss_fn(pred, conv, *target[i], i)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss
            
        return (giou_loss.numpy(), conf_loss.numpy(), prob_loss.numpy(), 
                total_loss.numpy())

    def train(self):
        
        loss_fn = YoloLoss(classes=self.model.classes)
        best_val_loss = 1000 # should be large at start
        for epoch in range(self.epochs):
            for image_data, target in self.trainset:
                results = self.train_step(image_data, target,loss_fn)
                cur_step = results[0]%self.steps_per_epoch
                print("epoch:{:2.0f} step:{:5.0f}/{}, lr:{:.6f}," \
                      "giou_loss:{:7.2f}, conf_loss:{:7.2f}," \
                      "prob_loss:{:7.2f}, total_loss:{:7.2f}"
                    .format(epoch, cur_step, self.steps_per_epoch, results[1], 
                            results[2], results[3], results[4], results[5]))

            if len(self.testset) == 0:
                print("configure TEST options to validate model")
                self.model.save_weights(
                    os.path.join(self.checkpointdir, self.model.name))
                continue
            
            count, giou_val, conf_val, prob_val, total_val = 0., 0, 0, 0, 0
            for image_data, target in self.testset:
                results = self.validate_step(image_data, target,loss_fn)
                count += 1
                giou_val += results[0]
                conf_val += results[1]
                prob_val += results[2]
                total_val += results[3]
            # writing validate summary data
            with self.validate_writer.as_default():
                tf.summary.scalar(
                    "validate_loss/total_val", total_val/count, step=epoch)
                tf.summary.scalar(
                    "validate_loss/giou_val", giou_val/count, step=epoch)
                tf.summary.scalar(
                    "validate_loss/conf_val", conf_val/count, step=epoch)
                tf.summary.scalar(
                    "validate_loss/prob_val", prob_val/count, step=epoch)
            self.validate_writer.flush()
                
            print("\n\ngiou_val_loss:{:7.2f}, conf_val_loss:{:7.2f}," \
                "prob_val_loss:{:7.2f}, total_val_loss:{:7.2f}\n\n".
                format(giou_val/count, conf_val/count, 
                       prob_val/count, total_val/count))

            if self.save_checkpoints and not self.save_best_checkpoint_only:
                save_directory = os.path.join(
                    self.checkpointdir, self.model.name+"_val_loss_{:7.2f}"
                    .format(total_val/count))
                self.model.save_weights(save_directory)
            if self.save_best_checkpoint_only and best_val_loss>total_val/count:
                save_directory = os.path.join(
                    self.checkpointdir, self.model.name)
                self.model.save_weights(save_directory)
                best_val_loss = total_val/count
            if not self.save_best_checkpoint_only and not self.save_checkpoints:
                save_directory = os.path.join(
                    self.checkpointdir, self.model.name)
                self.model.save_weights(save_directory)
