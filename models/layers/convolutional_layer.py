import tensorflow as tf

def convolutional_layer(name, inputs, filters: int, kernel_size: int, downsample: bool, batch_norm: bool, activation: str):
    """
    An implementation of the yolov3 custom convolutional layer. 

    Parameters
    ----------
    name : string
        The name of the tensor to be used in TensorBoard.
    inputs: tensor
        The output tensor of the previous convolutional layer.
    filters : int 
        The dimensionality of the output space (i.e. the number  
        of filters in the convolution).  
    kernel_size: int
        An integer or tuple/list of 2 integers, specifying the strides 
        of the convolution along the height and width.  
    downsample : bool
        Whether the input should be downsampled by a factor of two. If the input
        tensor is originally of shape:
        [batch_size, h, w, c]
        and downsampling is used, the resulting tensor will be of shape:
        [batch_size, h/2, w/2, c]
    batch_norm : bool
        Whether or not batch normalization should be applied to the input.
    activation : string
        If the activation is specified to be 'LEAKY' then a leaky relu activation will be
        used on the input. If no activation is specified, no activation will be used.

    Returns
    -------
    inputs : tensor
        The resulting tensor.
    shortcut : tensor
        A shortcut for a future residual block to connect to.
    """
    with tf.variable_scope(name):

        if batch_norm:
            use_bias = False
        else:
            use_bias = True

        if downsample:
            paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
            inputs = tf.pad(inputs, paddings, 'CONSTANT')
            strides = (2,2)
            padding = 'VALID'
        else:
            strides = (1,1)
            padding = 'SAME'

        inputs = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, use_bias=use_bias, padding=padding)
            
        if batch_norm:
            inputs = tf.layers.batch_normalization(inputs=inputs, momentum=0.9, epsilon=1e-05)
        
        if activation == 'LEAKY':
            inputs = tf.nn.leaky_relu(features=inputs, alpha=0.1)
        
        shortcut = inputs

        return inputs, shortcut