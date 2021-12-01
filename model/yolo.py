'''
Implements Yolo v4 model
Original Darknet implementation: https://github.com/AlexeyAB/darknet
'''

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import \
    Layer,Conv2D, LeakyReLU, ZeroPadding2D, BatchNormalization
from tensorflow.keras.regularizers import l2

def read_class_names(class_file_name):
    # loads class name from a file
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

class BatchNormalization(BatchNormalization):
    # "Frozen state" and "inference mode" are two separate concepts.
    # `layer.trainable = False` is to freeze the layer, so the layer will use
    # stored moving `var` and `mean` in the "inference mode", and both `gama`
    # and `beta` will not be updated !
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)

class Mish(Layer):
    def call(self,x):
        return x * tf.math.tanh(tf.math.softplus(x))

class Conv(Layer):
    '''
    Subclass of Keras layer
    Implements a connection: - Conv2D - BN - LeakyReLU -> 

    Args:
        filters (int):       filter of Conv2D
        kernel_size (int):   kernel of Conv2D
        strides (int):       stirdes of Conv2D
        downsample (bool):   if true, downsampling will be applied 
                             (sets strides = 2 and applies zero padding)
        activate (bool):     if true, the activation function in activate_type 
                             is used
        bn (str):            if true, batch normalization is performed
        activate_type (str): activation layer type. 
                             Either "mish" or "leakyReLU"
    '''

    def __init__(self, filters, kernel_size, downsample=False, activate=True, 
                 bn=True,activate_type='leakyReLU'):

        super(Conv, self).__init__()  
        self.downsample = downsample
        self.activate = activate
        self.bn = bn
        self.activate_type = activate_type
        self.filters = filters
        self.kernel_size = kernel_size

        if downsample:
            self.zp = ZeroPadding2D(((1, 0), (1, 0)))
            padding = 'valid'
            strides = 2
        else:
            strides = 1
            padding = 'same'

        self.conv = Conv2D(
            filters=self.filters, kernel_size = self.kernel_size, 
            strides=strides, padding=padding, use_bias=not bn, 
            kernel_regularizer=l2(0.0005), 
            kernel_initializer=tf.random_normal_initializer(stddev=0.01),
            bias_initializer=tf.constant_initializer(0.))
        if bn:
            self.batch_norm = BatchNormalization()
        if activate == True:
            if activate_type == "leakyReLU":
                self.leaky_re_lu = LeakyReLU(alpha=0.1)
            elif activate_type == "mish":
                self.mish = Mish()

    def call(self, x):
        if self.downsample:
            x = self.zp(x)
        x = self.conv(x)
        if self.bn:
            x = self.batch_norm(x)
        if self.activate == True:
            if self.activate_type == "leakyReLU":
                x = self.leaky_re_lu(x)
            elif self.activate_type == "mish":
                x = self.mish(x)
        return x

class ResidualBlock(Layer):
    '''
    Subclass of Keras layer
    Implements a residual block:  - Conv1 - Conv2 - + ->
                                  |_________________|       

    Args:
        filter_1,2 (int):    filter size passed to of conv1, conv2       
        activate_type (str): activation layer type passed to Conv. 
                             Either "mish" or "leakyReLU"
    '''

    def __init__(self, filter_1, filter_2, activate_type="leakyReLU"):
        super(ResidualBlock, self).__init__()      
        self.conv1 = Conv(
            filters = filter_1, kernel_size = 1, activate_type = activate_type)
        self.conv2 = Conv(
            filters = filter_2, kernel_size = 3, activate_type = activate_type)

    def call(self,x):
        y = self.conv1(x)
        y = self.conv2(y)
        return x+y

class Upsample(Layer):
    def __init__(self, method='nearest'):
        super(Upsample, self).__init__()
        self.method = method
    def call(self,x):  
        return tf.image.resize(
            x, (x.shape[1] * 2, x.shape[2] * 2), method=self.method)

class Decode(Layer):
    '''
    Subclass of Keras layer
    Implements a Decoding layer  
    adapted from: https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3

    Args:
        num_classes (int):   number of classes in the dataset
        i (int ):            0,1,2: grid scales
                             
    '''

    def __init__(self, num_classes, i=0, strides = [8, 16, 32], 
                 anchors = [[[12,  16], [19,   36], [40,   28]],
                           [[36,  75], [76,   55], [72,  146]],
                           [[142,110], [192, 243], [459, 401]]]):

        super(Decode, self).__init__()
        # where i = 0, 1 or 2 to correspond to the three grid scales  
        self.num_classes = num_classes
        self.i = i
        self.yolo_strides = np.array(strides)
        self.yolo_anchors = (np.array(anchors).T/strides).T

    def call(self, x):

        conv_shape       = tf.shape(x)
        batch_size       = conv_shape[0]
        output_size      = conv_shape[1]
        conv_output = tf.reshape(
            x, (batch_size, output_size, output_size, 3, 5 + self.num_classes))

        # offset of center position     
        conv_raw_dxdy = conv_output[:, :, :, :, 0:2] 
        # Prediction box length and width offset
        conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
        # confidence of the prediction box
        conv_raw_conf = conv_output[:, :, :, :, 4:5]
        # category probability of the prediction box 
        conv_raw_prob = conv_output[:, :, :, :, 5: ]

        # next need Draw the grid. Where output_size is equal to 13, 26 or 52  
        y = tf.range(output_size, dtype=tf.int32)
        y = tf.expand_dims(y, -1)
        y = tf.tile(y, [1, output_size])
        x = tf.range(output_size,dtype=tf.int32)
        x = tf.expand_dims(x, 0)
        x = tf.tile(x, [output_size, 1])

        xy_grid = tf.concat(
            [x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_grid = tf.tile(
            xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)

        # Calculate the center position of the prediction box:
        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) \
            * self.yolo_strides[self.i]
        # Calculate the length and width of the prediction box:
        pred_wh = (tf.exp(conv_raw_dwdh) * self.yolo_anchors[self.i]) \
            * self.yolo_strides[self.i]

        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
        # object box calculates the predicted confidence
        pred_conf = tf.sigmoid(conv_raw_conf) 
        # calculating the predicted probability category box object
        pred_prob = tf.sigmoid(conv_raw_prob) 

        # calculating the predicted probability category box object
        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

class CSPDarknet53(Layer):
    '''
    Subclass of Keras layer
    Implements a CSP Darknet https://github.com/AlexeyAB/darknet                            
    '''
    def __init__(self):
        super(CSPDarknet53, self).__init__()  

        self.conv1 = Conv(filters = 32, kernel_size = 3, activate_type="mish")

        # CSP 0
        self.conv2 = Conv(filters = 64, kernel_size = 3, downsample=True, 
                          activate_type="mish")
        self.conv3 = Conv(filters = 64, kernel_size = 1, activate_type="mish")
        self.conv4 = Conv(filters = 64, kernel_size = 1, activate_type="mish")
        #self.res1 = ResidualBlock(32, 64, activate_type="mish")
        self.conv5 = Conv(filters = 32, kernel_size = 1, activate_type="mish")
        self.conv6 = Conv(filters = 64, kernel_size = 3, activate_type="mish")
        self.conv7 = Conv(filters = 64, kernel_size = 1, activate_type="mish")
        # concatenate here
        self.conv8 = Conv(filters = 64, kernel_size = 1, activate_type="mish")
        
        ## CSP 1
        self.conv9 = Conv(filters = 128, kernel_size = 3, downsample=True, 
                          activate_type="mish")
        self.conv10 = Conv(filters = 64, kernel_size = 1, activate_type="mish")
        self.conv11 = Conv(filters = 64, kernel_size = 1, activate_type="mish")
        self.res1 = ResidualBlock(64, 64, activate_type="mish")
        self.res2 = ResidualBlock(64, 64, activate_type="mish")
        self.conv12 = Conv(filters = 64, kernel_size = 1, activate_type="mish")
        # concatenate here
        self.conv13 = Conv(filters = 128, kernel_size = 1, activate_type="mish")

        ## CSP 2
        self.conv14 = Conv(filters = 256, kernel_size = 3, downsample=True, 
                           activate_type="mish")
        self.conv15 = Conv(filters = 128, kernel_size = 1, activate_type="mish")
        self.conv16 = Conv(filters = 128, kernel_size = 1, activate_type="mish")
        self.res3 = ResidualBlock(128, 128, activate_type="mish")
        self.res4 = ResidualBlock(128, 128, activate_type="mish")
        self.res5 = ResidualBlock(128, 128, activate_type="mish")
        self.res6 = ResidualBlock(128, 128, activate_type="mish")
        self.res7 = ResidualBlock(128, 128, activate_type="mish")
        self.res8 = ResidualBlock(128, 128, activate_type="mish")
        self.res9 = ResidualBlock(128, 128, activate_type="mish")
        self.res10 = ResidualBlock(128, 128, activate_type="mish")
        self.conv17 = Conv(filters = 128, kernel_size = 1, activate_type="mish")
        # concatenate here
        self.conv18 = Conv(filters = 256, kernel_size = 1, activate_type="mish")

        ## CSP 3
        self.conv19 = Conv(filters = 512, kernel_size = 3, downsample=True, 
                           activate_type="mish")
        self.conv20 = Conv(filters = 256, kernel_size = 1, activate_type="mish")
        self.conv21 = Conv(filters = 256, kernel_size = 1, activate_type="mish")
        self.res11 = ResidualBlock(256, 256, activate_type="mish")
        self.res12 = ResidualBlock(256, 256, activate_type="mish")
        self.res13 = ResidualBlock(256, 256, activate_type="mish")
        self.res14 = ResidualBlock(256, 256, activate_type="mish")
        self.res15 = ResidualBlock(256, 256, activate_type="mish")
        self.res16 = ResidualBlock(256, 256, activate_type="mish")
        self.res17 = ResidualBlock(256, 256, activate_type="mish")
        self.res18 = ResidualBlock(256, 256, activate_type="mish")
        self.conv22 = Conv(filters = 256, kernel_size = 1, activate_type="mish")
        # concatenate here
        self.conv23 = Conv(filters = 512, kernel_size = 1, activate_type="mish")

        ## CSP 3
        self.conv24 = Conv(filters = 1024, kernel_size = 3, downsample=True, 
                           activate_type="mish")
        self.conv25 = Conv(filters = 512, kernel_size = 1, activate_type="mish")
        self.conv26 = Conv(filters = 512, kernel_size = 1, activate_type="mish")
        self.res19 = ResidualBlock(512, 512, activate_type="mish")
        self.res20 = ResidualBlock(512, 512, activate_type="mish")
        self.res21 = ResidualBlock(512, 512, activate_type="mish")
        self.res22 = ResidualBlock(512, 512, activate_type="mish")
        self.conv27 = Conv(filters = 512, kernel_size = 1, activate_type="mish")
        # concatenate here
        self.conv28 = Conv(filters = 1024, kernel_size = 1, activate_type="mish")

    def call(self,x):

        x = self.conv1(x)

        #CSP 0
        x = self.conv2(x)
        route = self.conv3(x)
        shortcut = self.conv4(x)
        x = self.conv5(shortcut)
        x = self.conv6(x)
        x = x + shortcut
        x = self.conv7(x)
        x = tf.concat([x, route], axis=-1)
        x = self.conv8(x)

        #CSP 1
        x = self.conv9(x)
        route = self.conv10(x)
        x = self.conv11(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.conv12(x)
        x = tf.concat([x, route], axis=-1)
        x = self.conv13(x)

        #CSP 2
        x = self.conv14(x)
        route = self.conv15(x)
        x = self.conv16(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        x = self.res7(x)
        x = self.res8(x)
        x = self.res9(x)
        x = self.res10(x)
        x = self.conv17(x)
        x = tf.concat([x, route], axis=-1)
        output_1 = self.conv18(x) 

        #CSP 3
        x = self.conv19(output_1)
        route = self.conv20(x)
        x = self.conv21(x)
        x = self.res11(x)
        x = self.res12(x)
        x = self.res13(x)
        x = self.res14(x)
        x = self.res15(x)
        x = self.res16(x)
        x = self.res17(x)
        x = self.res18(x)
        x = self.conv22(x)
        x = tf.concat([x, route], axis=-1)
        output_2 = self.conv23(x)

        #CSP 4
        x = self.conv24(output_2)
        route = self.conv25(x)
        x = self.conv26(x)
        x = self.res19(x)
        x = self.res20(x)
        x = self.res21(x)
        x = self.res22(x)
        x = self.conv27(x)
        x = tf.concat([x, route], axis=-1)
        output_3 = self.conv28(x)

        return output_1, output_2, output_3
        #return route_1, route_2, x

class YOLOv4(Model):
    '''
    Subclass of Keras layer
    Implements YOLOv4 with backbone (CSPDarknet53), neck and head
    '''

    def __init__(self,classes):
        super(YOLOv4, self).__init__()
        self.classes = classes
        self.num_classes = len(read_class_names(classes))

        self.cspdarknet53 = CSPDarknet53()

        ## NECK
        self.conv1 = Conv(filters = 512, kernel_size = 1)
        self.conv2 = Conv(filters = 1024, kernel_size = 3)
        self.conv3 = Conv(filters = 512, kernel_size = 1)

        ### SPP ###
        self.max_pooling_1 = tf.keras.layers.MaxPool2D(
            pool_size=13, padding='SAME', strides=1)
        self.max_pooling_2 = tf.keras.layers.MaxPool2D(
            pool_size=9, padding='SAME', strides=1)
        self.max_pooling_3 = tf.keras.layers.MaxPool2D(
            pool_size=5, padding='SAME', strides=1)
        ### End SPP ###

        self.conv4 = Conv(filters = 512, kernel_size = 1)
        self.conv5 = Conv(filters = 1024, kernel_size = 3)
        self.conv6 = Conv(filters = 512, kernel_size = 1)
        self.conv7 = Conv(filters = 256, kernel_size = 1)
        self.conv8 = Conv(filters = 256, kernel_size = 1)
        self.up1 = Upsample()
        self.conv9 = Conv(filters = 256, kernel_size = 1)
        self.conv10 = Conv(filters = 512, kernel_size = 3)
        self.conv11 = Conv(filters = 256, kernel_size = 1)
        self.conv12 = Conv(filters = 512, kernel_size = 3)
        self.conv13 = Conv(filters = 256, kernel_size = 1)
        self.conv14 = Conv(filters = 128, kernel_size = 1)
        self.up2 = Upsample()
        self.conv15 = Conv(filters = 128, kernel_size = 1)
        self.conv16 = Conv(filters = 128, kernel_size = 1)
        self.conv17 = Conv(filters = 256, kernel_size = 3)
        self.conv18 = Conv(filters = 128, kernel_size = 1)
        self.conv19 = Conv(filters = 256, kernel_size = 3)
        self.conv20 = Conv(filters = 128, kernel_size = 1)
        
        ## HEAD 
        self.conv21 = Conv(filters = 256, kernel_size = 3)
        self.conv22 = Conv(filters = 3 * (self.num_classes + 5), 
                           kernel_size = 1, activate=False, bn=False)

        self.conv23 = Conv(filters = 256, kernel_size = 3, downsample=True)

        self.conv24 = Conv(filters = 256, kernel_size = 1)
        self.conv25 = Conv(filters = 512, kernel_size = 3)
        self.conv26 = Conv(filters = 256, kernel_size = 1)
        self.conv27 = Conv(filters = 512, kernel_size = 3)
        self.conv28 = Conv(filters = 256, kernel_size = 1)

        self.conv29 = Conv(filters = 512, kernel_size = 3)
        self.conv30 = Conv(filters = 3 * (self.num_classes + 5), 
                           kernel_size = 1, activate=False, bn=False)

        self.conv31 = Conv(filters = 512, kernel_size = 3, downsample=True)

        self.conv32 = Conv(filters = 512, kernel_size = 1)
        self.conv33 = Conv(filters = 1024, kernel_size = 3)
        self.conv34 = Conv(filters = 512, kernel_size = 1)
        self.conv35 = Conv(filters = 1024, kernel_size = 3)
        self.conv36 = Conv(filters = 512, kernel_size = 1)

        self.conv37 = Conv(filters = 1024, kernel_size = 3)
        self.conv38 = Conv(filters = 3 * (self.num_classes + 5), 
                           kernel_size = 1, activate=False, bn=False)

        self.decode1 = Decode(self.num_classes, 0)
        self.decode2 = Decode(self.num_classes, 1)
        self.decode3 = Decode(self.num_classes, 2)

    def call(self,x):
        ## BACKBONE
        input_1, input_2, input_3 = self.cspdarknet53(x)

        # NECK
        x = self.conv1(input_3)
        x = self.conv2(x)
        x = self.conv3(x)

        max_pooling_1 = self.max_pooling_1(x)
        max_pooling_2 = self.max_pooling_2(x)
        max_pooling_3 = self.max_pooling_3(x)

        spp = tf.concat(
            [max_pooling_1, max_pooling_2, max_pooling_3, x], axis=-1)

        x = self.conv4(spp)
        x = self.conv5(x)
        output_3 = self.conv6(x)
        x = self.conv7(output_3)
        upsampled = self.up1(x)

        x = self.conv8(input_2)
        x = tf.concat([x,upsampled], axis=-1)

        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        output_2 = self.conv13(x)
        x = self.conv14(output_2)
        upsampled = self.up2(x)

        x = self.conv15(input_1)
        x = tf.concat([x,upsampled], axis=-1)

        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        x = self.conv19(x)
        output_1 = self.conv20(x)

        ## HEAD 
        x = self.conv21(output_1)
        route_1 = self.conv22(x)

        x = self.conv23(output_1)
        x = tf.concat([x, output_2], axis=-1)

        x = self.conv24(x)
        x = self.conv25(x)
        x = self.conv26(x)
        x = self.conv27(x)
        connection = self.conv28(x)

        x = self.conv29(connection)
        route_2 = self.conv30(x)

        x = self.conv31(connection)
        x = tf.concat([x, output_3], axis=-1)

        x = self.conv32(x)
        x = self.conv33(x)
        x = self.conv34(x)
        x = self.conv35(x)
        x = self.conv36(x)

        x = self.conv37(x)
        route_3 = self.conv38(x)
        
        conv_tensors = [route_1, route_2, route_3]
        
        pred_tensor1 = self.decode1(conv_tensors[0])
        pred_tensor2 = self.decode2(conv_tensors[1])
        pred_tensor3 = self.decode3(conv_tensors[2])
        #if training: 
        output_tensors = [conv_tensors[0],pred_tensor1,
                            conv_tensors[1],pred_tensor2,
                            conv_tensors[2],pred_tensor3]
        # else: 
        #     output_tensors = [pred_tensor1,pred_tensor2,pred_tensor3]
        return output_tensors



def bbox_iou(boxes1, boxes2):
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    return 1.0 * inter_area / union_area

def bbox_giou(boxes1, boxes2):
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) \
        * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) \
        * (boxes2[..., 3] - boxes2[..., 1])

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    # Calculate the iou value between the two bounding boxes
    iou = inter_area / union_area

    # Calculate the coordinates of the upper left corner and the lower right 
    # corner of the smallest closed convex surface
    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)

    # Calculate the area of the smallest closed convex surface C
    enclose_area = enclose[..., 0] * enclose[..., 1]

    # Calculate the GIoU value according to the GioU formula  
    giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

    return giou

def bbox_ciou(boxes1, boxes2):
    boxes1_coor = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2_coor = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left = tf.maximum(boxes1_coor[..., 0], boxes2_coor[..., 0])
    up = tf.maximum(boxes1_coor[..., 1], boxes2_coor[..., 1])
    right = tf.maximum(boxes1_coor[..., 2], boxes2_coor[..., 2])
    down = tf.maximum(boxes1_coor[..., 3], boxes2_coor[..., 3])

    c = (right - left) * (right - left) + (up - down) * (up - down)
    iou = bbox_iou(boxes1, boxes2)

    u = (boxes1[..., 0] - boxes2[..., 0]) * (boxes1[..., 0] - boxes2[..., 0]) \
        + (boxes1[..., 1] - boxes2[..., 1]) * (boxes1[..., 1] - boxes2[..., 1])
    d = u / c

    ar_gt = boxes2[..., 2] / boxes2[..., 3]
    ar_pred = boxes1[..., 2] / boxes1[..., 3]

    ar_loss = 4 / (np.pi * np.pi) * (tf.atan(ar_gt) - tf.atan(ar_pred)) \
        * (tf.atan(ar_gt) - tf.atan(ar_pred))
    alpha = ar_loss / (1 - iou + ar_loss + 0.000001)
    ciou_term = d + alpha * ar_loss

    return iou - ciou_term

class YoloLoss():
    '''
    implements yolo loss object
    '''

    def __init__(self, classes, yolo_strides = [8, 16, 32], iou_loss_thresh = 0.5):
        super(YoloLoss, self).__init__()
        self.num_classes = len(read_class_names(classes))
        self.yolo_strides = yolo_strides
        self.iou_loss_thresh = iou_loss_thresh
    def __call__(self, pred, conv, label, bboxes, i=0):
        conv_shape  = tf.shape(conv)
        batch_size  = conv_shape[0]
        output_size = conv_shape[1]
        input_size  = self.yolo_strides[i] * output_size
        conv = tf.reshape(
            conv, (batch_size, output_size, output_size, 3, 5 + self.num_classes))

        conv_raw_conf = conv[:, :, :, :, 4:5]
        conv_raw_prob = conv[:, :, :, :, 5:]

        pred_xywh     = pred[:, :, :, :, 0:4]
        pred_conf     = pred[:, :, :, :, 4:5]

        label_xywh    = label[:, :, :, :, 0:4]
        respond_bbox  = label[:, :, :, :, 4:5]
        label_prob    = label[:, :, :, :, 5:]

        giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)
        input_size = tf.cast(input_size, tf.float32)

        bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] \
            * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
        giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

        iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], 
                    bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
        # Find the value of IoU with the real box The largest prediction box
        max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

        # If the largest iou is less than the threshold, it is considered that 
        # the prediction box contains no objects, then the background box
        respond_bgd = (1.0 - respond_bbox) \
            * tf.cast( max_iou < self.iou_loss_thresh, tf.float32 )

        conf_focal = tf.pow(respond_bbox - pred_conf, 2)

        # Calculate the loss of confidence
        # we hope that if the grid contains objects, then the network output 
        # prediction box has a confidence of 1 and 0 when there is no object.
        conf_loss = conf_focal * (
                respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=respond_bbox, logits=conv_raw_conf)
                +
                respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=respond_bbox, logits=conv_raw_conf)
        )

        prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(
            labels=label_prob, logits=conv_raw_prob)

        giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4]))
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
        prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4]))

        return giou_loss, conf_loss, prob_loss