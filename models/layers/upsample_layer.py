import tensorflow as tf

def upsample_layer(name, inputs):
    """
    Takes the outputs of the previous convolutional layer and upsamples them by a factor of two
    using the 'nearest neighbor' method.

    Parameters
    ----------
    name : string
        The name of the tensor to be used in TensorBoard.
    inputs : tensor
        The output of the previous convolutional layer. 
        This tensor will have the shape of:
        [batch_size, h, w, c]

    Returns
    -------
    inputs : tensor
        A tensor of shape:
        [batch_size, 2 * h, 2 * w, c]
    """

    with tf.variable_scope(name):

        inputs = tf.image.resize_nearest_neighbor(inputs, (inputs.shape[1]*2, inputs.shape[2]*2))
    
    return inputs
