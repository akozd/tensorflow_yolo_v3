import tensorflow as tf
from models.layers.convolutional_layer import convolutional_layer
from models.layers.route_layer import route_layer
from models.layers.shortcut_layer import shortcut_layer
from models.layers.upsample_layer import upsample_layer
from models.layers.yolo_layer import yolo_layer

def yolo_v3(inputs, num_classes, anchors, h=416, w=416, training=False):
    """
    The fully defined architecture of the version three yolo model. 

    Parameters
    ----------
    inputs : tensor
        An array of shape:
        [batch_size, image_height, image_width, 3]
        The inputs here are a batch of images to train on or to detect objects in.
        The images are loaded in 'RGB' and divided by 255.0 so that the values of 
        the array are between 0 and 1.
    num_classes : int
        The number of classes in the training data.
    anchors : list
        A list of anchors with format:
        [[anchor1_width, anchor1_height], [anchor2_width, anchor2_height], [anchor3_width, anchor3_height], ...]
        These anchors are needed in the yolo layers to convert the raw box predictions up to the scale of the
        original image.
    h : int
        The original height of the input image.
    w : int
        The original width of the input image.
    training : bool
        Whether or not the model is currently training. If it is not, the model only needs to return
        the output from its yolo layers. However, if the model is in fact training, it needs to 
        return the outputs of its convolutional layers that are immediately before the yolo layers.
        These raw convolutional outputs are necessary for the computation of the loss function.

    Returns
    -------
    large_object_box_detections : tensor
        Contains the detectors responsible for predicting the bounding boxes of large objects. 
        Tensor of shape:
        [batch_size, large_yolo_layer_grid_h, large_yolo_layer_grid_w, num_anchors_per_layer, 5 + num_classes]

    medium_object_box_detections : tensor
        Contains the detectors responsible for predicting the bounding boxes of medium objects. 
        Tensor of shape:
        [batch_size, med_yolo_layer_grid_h, med_yolo_layer_grid_w, num_anchors_per_layer, 5 + num_classes]
    
    small_object_box_detections : tensor
        Contains the detectors responsible for predicting the bounding boxes of small objects. 
        Tensor of shape:
        [batch_size, small_yolo_layer_grid_h, small_yolo_layer_grid_w, num_anchors_per_layer, 5 + num_classes]

    large_object_raw_detections : tensor
        Contains the raw outputs of the convolutional layer that will feed into the yolo layer
        responsible for detecting large objects.
        Tensor of shape:
        [batch_size, large_yolo_layer_grid_h, large_yolo_layer_grid_w, num_anchors_per_layer*(5 + num_classes)]
    
    medium_object_raw_detections : tensor
        Contains the raw outputs of the convolutional layer that will feed into the yolo layer
        responsible for detecting medium objects.
        Tensor of shape:
        [batch_size, med_yolo_layer_grid_h, med_yolo_layer_grid_w, num_anchors_per_layer*(5 + num_classes)]
    
    small_object_raw_detections : tensor
        Contains the raw outputs of the convolutional layer that will feed into the yolo layer
        responsible for detecting small objects.
        Tensor of shape:
        [batch_size, small_yolo_layer_grid_h, small_yolo_layer_grid_w, num_anchors_per_layer*(5 + num_classes)]
    """ 
    
    with tf.variable_scope('darknet'):

        inputs, _ = convolutional_layer(name='conv_0', inputs=inputs, filters=32, kernel_size=3, downsample=False, batch_norm=True, activation='LEAKY')
        inputs, shortcut = convolutional_layer(name='conv_1', inputs=inputs, filters=64, kernel_size=3, downsample=True, batch_norm=True, activation='LEAKY')
        inputs, _ = convolutional_layer(name='conv_2', inputs=inputs, filters=32, kernel_size=1, downsample=False, batch_norm=True, activation='LEAKY')
        inputs, _ = convolutional_layer(name='conv_3', inputs=inputs, filters=64, kernel_size=3, downsample=False, batch_norm=True, activation='LEAKY')
        inputs, _ = shortcut_layer(name='shortcut_0', shortcut=shortcut, inputs=inputs)
        inputs, shortcut = convolutional_layer(name='conv_4', inputs=inputs, filters=128, kernel_size=3, downsample=True, batch_norm=True, activation='LEAKY')
        
        inputs, _ = convolutional_layer(name='conv_5', inputs=inputs, filters=64, kernel_size=1, downsample=False, batch_norm=True, activation='LEAKY')
        inputs, _ = convolutional_layer(name='conv_6', inputs=inputs, filters=128, kernel_size=3, downsample=False, batch_norm=True, activation='LEAKY')
        inputs, shortcut = shortcut_layer(name='shortcut_1', shortcut=shortcut, inputs=inputs)
        
        inputs, _ = convolutional_layer(name='conv_7', inputs=inputs, filters=64, kernel_size=1, downsample=False, batch_norm=True, activation='LEAKY')
        inputs, _ = convolutional_layer(name='conv_8', inputs=inputs, filters=128, kernel_size=3, downsample=False, batch_norm=True, activation='LEAKY')
        inputs, _ = shortcut_layer(name='shortcut_2', shortcut=shortcut, inputs=inputs)

        inputs, shortcut = convolutional_layer(name='conv_9', inputs=inputs, filters=256, kernel_size=3, downsample=True, batch_norm=True, activation='LEAKY')
        
        inputs, _ = convolutional_layer(name='conv_10', inputs=inputs, filters=128, kernel_size=1, downsample=False, batch_norm=True, activation='LEAKY')
        inputs, _ = convolutional_layer(name='conv_11', inputs=inputs, filters=256, kernel_size=3, downsample=False, batch_norm=True, activation='LEAKY')
        inputs, shortcut = shortcut_layer(name='shortcut_3', shortcut=shortcut, inputs=inputs)
        
        inputs, _ = convolutional_layer(name='conv_12', inputs=inputs, filters=128, kernel_size=1, downsample=False, batch_norm=True, activation='LEAKY')
        inputs, _ = convolutional_layer(name='conv_13', inputs=inputs, filters=256, kernel_size=3, downsample=False, batch_norm=True, activation='LEAKY')
        inputs, shortcut = shortcut_layer(name='shortcut_4', shortcut=shortcut, inputs=inputs)
        
        inputs, _ = convolutional_layer(name='conv_14', inputs=inputs, filters=128, kernel_size=1, downsample=False, batch_norm=True, activation='LEAKY')
        inputs, _ = convolutional_layer(name='conv_15', inputs=inputs, filters=256, kernel_size=3, downsample=False, batch_norm=True, activation='LEAKY')
        inputs, shortcut = shortcut_layer(name='shortcut_5', shortcut=shortcut, inputs=inputs)
        
        inputs, _ = convolutional_layer(name='conv_16', inputs=inputs, filters=128, kernel_size=1, downsample=False, batch_norm=True, activation='LEAKY')
        inputs, _ = convolutional_layer(name='conv_17', inputs=inputs, filters=256, kernel_size=3, downsample=False, batch_norm=True, activation='LEAKY')
        inputs, shortcut = shortcut_layer(name='shortcut_6', shortcut=shortcut, inputs=inputs)

        inputs, _ = convolutional_layer(name='conv_18', inputs=inputs, filters=128, kernel_size=1, downsample=False, batch_norm=True, activation='LEAKY')
        inputs, _ = convolutional_layer(name='conv_19', inputs=inputs, filters=256, kernel_size=3, downsample=False, batch_norm=True, activation='LEAKY')
        inputs, shortcut = shortcut_layer(name='shortcut_7', shortcut=shortcut, inputs=inputs)

        inputs, _ = convolutional_layer(name='conv_20', inputs=inputs, filters=128, kernel_size=1, downsample=False, batch_norm=True, activation='LEAKY')
        inputs, _ = convolutional_layer(name='conv_21', inputs=inputs, filters=256, kernel_size=3, downsample=False, batch_norm=True, activation='LEAKY')
        inputs, shortcut = shortcut_layer(name='shortcut_8', shortcut=shortcut, inputs=inputs)

        inputs, _ = convolutional_layer(name='conv_22', inputs=inputs, filters=128, kernel_size=1, downsample=False, batch_norm=True, activation='LEAKY')
        inputs, _ = convolutional_layer(name='conv_23', inputs=inputs, filters=256, kernel_size=3, downsample=False, batch_norm=True, activation='LEAKY')
        inputs, shortcut = shortcut_layer(name='shortcut_9', shortcut=shortcut, inputs=inputs)

        inputs, _ = convolutional_layer(name='conv_24', inputs=inputs, filters=128, kernel_size=1, downsample=False, batch_norm=True, activation='LEAKY')
        inputs, _ = convolutional_layer(name='conv_25', inputs=inputs, filters=256, kernel_size=3, downsample=False, batch_norm=True, activation='LEAKY')
        inputs, darknet_route_1 = shortcut_layer(name='shortcut_10', shortcut=shortcut, inputs=inputs)
        
        inputs, shortcut = convolutional_layer(name='conv_26', inputs=inputs, filters=512, kernel_size=3, downsample=True, batch_norm=True, activation='LEAKY')

        inputs, _ = convolutional_layer(name='conv_27', inputs=inputs, filters=256, kernel_size=1, downsample=False, batch_norm=True, activation='LEAKY')
        inputs, _ = convolutional_layer(name='conv_28', inputs=inputs, filters=512, kernel_size=3, downsample=False, batch_norm=True, activation='LEAKY')
        inputs, shortcut = shortcut_layer(name='shortcut_11', shortcut=shortcut, inputs=inputs)

        inputs, _ = convolutional_layer(name='conv_29', inputs=inputs, filters=256, kernel_size=1, downsample=False, batch_norm=True, activation='LEAKY')
        inputs, _ = convolutional_layer(name='conv_30', inputs=inputs, filters=512, kernel_size=3, downsample=False, batch_norm=True, activation='LEAKY')
        inputs, shortcut = shortcut_layer(name='shortcut_12', shortcut=shortcut, inputs=inputs)

        inputs, _ = convolutional_layer(name='conv_31', inputs=inputs, filters=256, kernel_size=1, downsample=False, batch_norm=True, activation='LEAKY')
        inputs, _ = convolutional_layer(name='conv_32', inputs=inputs, filters=512, kernel_size=3, downsample=False, batch_norm=True, activation='LEAKY')
        inputs, shortcut = shortcut_layer(name='shortcut_13', shortcut=shortcut, inputs=inputs)

        inputs, _ = convolutional_layer(name='conv_33', inputs=inputs, filters=256, kernel_size=1, downsample=False, batch_norm=True, activation='LEAKY')
        inputs, _ = convolutional_layer(name='conv_34', inputs=inputs, filters=512, kernel_size=3, downsample=False, batch_norm=True, activation='LEAKY')
        inputs, shortcut = shortcut_layer(name='shortcut_14', shortcut=shortcut, inputs=inputs)

        inputs, _ = convolutional_layer(name='conv_35', inputs=inputs, filters=256, kernel_size=1, downsample=False, batch_norm=True, activation='LEAKY')
        inputs, _ = convolutional_layer(name='conv_36', inputs=inputs, filters=512, kernel_size=3, downsample=False, batch_norm=True, activation='LEAKY')
        inputs, shortcut = shortcut_layer(name='shortcut_15', shortcut=shortcut, inputs=inputs)

        inputs, _ = convolutional_layer(name='conv_37', inputs=inputs, filters=256, kernel_size=1, downsample=False, batch_norm=True, activation='LEAKY')
        inputs, _ = convolutional_layer(name='conv_38', inputs=inputs, filters=512, kernel_size=3, downsample=False, batch_norm=True, activation='LEAKY')
        inputs, shortcut = shortcut_layer(name='shortcut_16', shortcut=shortcut, inputs=inputs)
        
        inputs, _ = convolutional_layer(name='conv_39', inputs=inputs, filters=256, kernel_size=1, downsample=False, batch_norm=True, activation='LEAKY')
        inputs, _ = convolutional_layer(name='conv_40', inputs=inputs, filters=512, kernel_size=3, downsample=False, batch_norm=True, activation='LEAKY')
        inputs, shortcut = shortcut_layer(name='shortcut_17',shortcut=shortcut, inputs=inputs)

        inputs, _ = convolutional_layer(name='conv_41', inputs=inputs, filters=256, kernel_size=1, downsample=False, batch_norm=True, activation='LEAKY')
        inputs, _ = convolutional_layer(name='conv_42', inputs=inputs, filters=512, kernel_size=3, downsample=False, batch_norm=True, activation='LEAKY')
        inputs, darknet_route_2 = shortcut_layer(name='shortcut_18',shortcut=shortcut, inputs=inputs)

        inputs, shortcut = convolutional_layer(name='conv_43', inputs=inputs, filters=1024, kernel_size=3, downsample=True, batch_norm=True, activation='LEAKY')

        inputs, _ = convolutional_layer(name='conv_44', inputs=inputs, filters=512, kernel_size=1, downsample=False, batch_norm=True, activation='LEAKY')
        inputs, _ = convolutional_layer(name='conv_45', inputs=inputs, filters=1024, kernel_size=3, downsample=False, batch_norm=True, activation='LEAKY')
        inputs, shortcut = shortcut_layer(name='shortcut_19', shortcut=shortcut, inputs=inputs)

        inputs, _ = convolutional_layer(name='conv_46', inputs=inputs, filters=512, kernel_size=1, downsample=False, batch_norm=True, activation='LEAKY')
        inputs, _ = convolutional_layer(name='conv_47', inputs=inputs, filters=1024, kernel_size=3, downsample=False, batch_norm=True, activation='LEAKY')
        inputs, shortcut = shortcut_layer(name='shortcut_20', shortcut=shortcut, inputs=inputs)

        inputs, _ = convolutional_layer(name='conv_48', inputs=inputs, filters=512, kernel_size=1, downsample=False, batch_norm=True, activation='LEAKY')
        inputs, _ = convolutional_layer(name='conv_49', inputs=inputs, filters=1024, kernel_size=3, downsample=False, batch_norm=True, activation='LEAKY')
        inputs, shortcut = shortcut_layer(name='shortcut_21', shortcut=shortcut, inputs=inputs)
    
        inputs, _ = convolutional_layer(name='conv_50', inputs=inputs, filters=512, kernel_size=1, downsample=False, batch_norm=True, activation='LEAKY')
        inputs, _ = convolutional_layer(name='conv_51', inputs=inputs, filters=1024, kernel_size=3, downsample=False, batch_norm=True, activation='LEAKY')
        inputs, _ = shortcut_layer(name='shortcut_22', shortcut=shortcut, inputs=inputs) 
    
    inputs, _ = convolutional_layer(name='conv_52', inputs=inputs, filters=512, kernel_size=1, downsample=False, batch_norm=True, activation='LEAKY')
    inputs, _ = convolutional_layer(name='conv_53', inputs=inputs, filters=1024, kernel_size=3, downsample=False, batch_norm=True, activation='LEAKY')
    inputs, _ = convolutional_layer(name='conv_54', inputs=inputs, filters=512, kernel_size=1, downsample=False, batch_norm=True, activation='LEAKY')
    inputs, _ = convolutional_layer(name='conv_55', inputs=inputs, filters=1024, kernel_size=3, downsample=False, batch_norm=True, activation='LEAKY')   
    inputs, yolo_route = convolutional_layer(name='conv_56', inputs=inputs, filters=512, kernel_size=1, downsample=False, batch_norm=True, activation='LEAKY')
    inputs, _ = convolutional_layer(name='conv_57', inputs=inputs, filters=1024, kernel_size=3, downsample=False, batch_norm=True, activation='LEAKY')

    large_object_raw_detections, _ = convolutional_layer(name='conv_58', inputs=inputs, filters=3*(num_classes+5), kernel_size=1, downsample=False, batch_norm=False, activation='LINEAR')
    large_object_box_detections = yolo_layer(name='yolo_0', inputs=large_object_raw_detections, image_height=h, image_width=w, num_classes=num_classes, anchors=anchors[6:9])

    inputs, _ = convolutional_layer(name='conv_59', inputs=yolo_route, filters=256, kernel_size=1, downsample=False, batch_norm=True, activation='LEAKY')
    inputs = upsample_layer(name='upsample_0', inputs=inputs)
    inputs = route_layer(name='route_0', inputs=inputs, route=darknet_route_2)

    inputs, _ = convolutional_layer(name='conv_60', inputs=inputs, filters=256, kernel_size=1, downsample=False, batch_norm=True, activation='LEAKY')
    inputs, _ = convolutional_layer(name='conv_61', inputs=inputs, filters=512, kernel_size=3, downsample=False, batch_norm=True, activation='LEAKY')
    inputs, _ = convolutional_layer(name='conv_62', inputs=inputs, filters=256, kernel_size=1, downsample=False, batch_norm=True, activation='LEAKY')
    inputs, _ = convolutional_layer(name='conv_63', inputs=inputs, filters=512, kernel_size=3, downsample=False, batch_norm=True, activation='LEAKY')   
    inputs, yolo_route = convolutional_layer(name='conv_64', inputs=inputs, filters=256, kernel_size=1, downsample=False, batch_norm=True, activation='LEAKY')
    inputs, _ = convolutional_layer(name='conv_65', inputs=inputs, filters=512, kernel_size=3, downsample=False, batch_norm=True, activation='LEAKY')

    medium_object_raw_detections, _ = convolutional_layer(name='conv_66', inputs=inputs, filters=3*(num_classes+5), kernel_size=1, downsample=False, batch_norm=False, activation='LINEAR')
    medium_object_box_detections = yolo_layer(name='yolo_1', inputs=medium_object_raw_detections, image_height=h, image_width=w, num_classes=num_classes, anchors=anchors[3:6])

    inputs, _ = convolutional_layer(name='conv_67', inputs=yolo_route, filters=128, kernel_size=1, downsample=False, batch_norm=True, activation='LEAKY')
    inputs = upsample_layer(name='upsample_1', inputs=inputs)
    inputs = route_layer(name='route_1', inputs=inputs, route=darknet_route_1)

    inputs, _ = convolutional_layer(name='conv_68', inputs=inputs, filters=128, kernel_size=1, downsample=False, batch_norm=True, activation='LEAKY')
    inputs, _ = convolutional_layer(name='conv_69', inputs=inputs, filters=256, kernel_size=3, downsample=False, batch_norm=True, activation='LEAKY')
    inputs, _ = convolutional_layer(name='conv_70', inputs=inputs, filters=128, kernel_size=1, downsample=False, batch_norm=True, activation='LEAKY')
    inputs, _ = convolutional_layer(name='conv_71', inputs=inputs, filters=256, kernel_size=3, downsample=False, batch_norm=True, activation='LEAKY')   
    inputs, _ = convolutional_layer(name='conv_72', inputs=inputs, filters=128, kernel_size=1, downsample=False, batch_norm=True, activation='LEAKY')
    inputs, _ = convolutional_layer(name='conv_73', inputs=inputs, filters=256, kernel_size=3, downsample=False, batch_norm=True, activation='LEAKY')

    small_object_raw_detections, _ = convolutional_layer(name='conv_74', inputs=inputs, filters=3*(num_classes+5), kernel_size=1, downsample=False, batch_norm=False, activation='LINEAR')
    small_object_box_detections = yolo_layer(name='yolo_2', inputs=small_object_raw_detections, image_height=h, image_width=w, num_classes=num_classes, anchors=anchors[0:3])
    
    if training == False:
        
        with tf.variable_scope('reshape_yolo_0'):
            large_object_box_detections = tf.reshape(large_object_box_detections, [-1, large_object_box_detections.shape[1]*large_object_box_detections.shape[2]*large_object_box_detections.shape[3], num_classes + 5])
        with tf.variable_scope('reshape_yolo_1'):
            medium_object_box_detections = tf.reshape(medium_object_box_detections, [-1, medium_object_box_detections.shape[1]*medium_object_box_detections.shape[2]*medium_object_box_detections.shape[3], num_classes + 5])
        with tf.variable_scope('reshape_yolo_2'):
            small_object_box_detections = tf.reshape(small_object_box_detections, [-1, small_object_box_detections.shape[1]*small_object_box_detections.shape[2]*small_object_box_detections.shape[3], num_classes + 5])
        
        return large_object_box_detections, medium_object_box_detections, small_object_box_detections
    else:
        return large_object_box_detections, medium_object_box_detections, small_object_box_detections, large_object_raw_detections, medium_object_raw_detections, small_object_raw_detections