import tensorflow as tf

def yolo_layer(name, inputs, anchors, num_classes, image_height, image_width):
    """
    Takes the output of a convolutional layer and transforms it into fully scaled
    bounding box coordinates. These predicted boxes can then be passed through a non-max suppression
    algorithm to eliminate significant overlap in boxes. The yolo layer is the final layer of
    the yolov3 model, and it is applied three times at varying scales in order to help with 
    the detection of large, medium, and small objects.

    Parameters
    ----------
    name : string
        The name of the tensor to be used in TensorBoard.
    inputs : tensor
        The output of the previous convolutional layer. 
        This tensor will have the shape of:
        [batch_size, yolo_layer_grid_h, yolo_layer_grid_w, num_anchors_per_layer * (5 + num_classes)]
        where the '5' represents the boxs' coordinates and object confidence score.
    anchors : list
        A sublist of the anchors list, of length num_anchors/num_layers. 
        The formatting of the sublist is as follows: 
        [[anchor1_width, anchor1_height], [anchor2_width, anchor2_height], [anchor3_width, anchor3_height]]
        The anchors work across all of the layer's detectors, acting as 'guides' for the 
        bounding box predictions. 
    num_classes : int
        The number of classes in the training data.
    image_height : int
        The height of the input image.
    image_width : int
        The width of the input image.

    Returns
    -------
    inputs : tensor
        A tensor of shape:
        [batch_size, yolo_layer_grid_h, yolo_layer_grid_w, num_anchors_per_layer, 5 + num_classes]
        The box coordinates are of the form:
        [center_x, center_y, width, height]
        and are fully scaled up to reflect the original dimensions of the input image.
    """

    with tf.variable_scope(name):

        inputs_shape = inputs.get_shape().as_list()
        stride_x = image_width // inputs_shape[2] 
        stride_y = image_height // inputs_shape[1]

        num_anchors = len(anchors)
        anchors = tf.constant([[a[0] / stride_x, a[1] / stride_y] for a in anchors], dtype=tf.float32) # convert to scale
        anchors_w = tf.reshape(anchors[:,0], [1,1,1,num_anchors,1])
        anchors_h = tf.reshape(anchors[:,1], [1,1,1,num_anchors,1])
        
        clustroid_x = tf.tile(tf.reshape(tf.range(inputs_shape[2], dtype=tf.float32), [1, -1, 1, 1]), [inputs_shape[2], 1, 1, 1])
        clustroid_y = tf.tile(tf.reshape(tf.range(inputs_shape[1], dtype=tf.float32), [-1, 1, 1, 1]), [1, inputs_shape[1], 1, 1])
        
        # [ ? x 13 x 13 x num_anchors x (5 + num_classes)]
        inputs = tf.reshape(inputs, [-1,inputs_shape[1], inputs_shape[2], num_anchors, num_classes + 5])
        delta_x, delta_y, delta_w, delta_h, obj_conf, class_conf = tf.split(inputs, [1, 1, 1, 1, 1, num_classes], axis=-1)

        # add grid offsets and multiply by stride to bring up to scale
        box_x = (clustroid_x + tf.nn.sigmoid(delta_x)) * stride_x
        box_y = (clustroid_y + tf.nn.sigmoid(delta_y)) * stride_y
        box_w = anchors_w * tf.exp(delta_w) * stride_x
        box_h = anchors_h * tf.exp(delta_h) * stride_y
        obj_conf = tf.nn.sigmoid(obj_conf)
        class_conf = tf.nn.sigmoid(class_conf)

        inputs = tf.concat([box_x, box_y, box_w, box_h, obj_conf, class_conf], axis=-1)

    return inputs
