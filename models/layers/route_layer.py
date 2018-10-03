import tensorflow as tf

def route_layer(name, inputs, route):
    """
    Takes the inputs tensor and route tensor and then concatenates them
    along the last axis (the channel axis).

    Parameters
    ----------
    name : string
        The name of the tensor to be used in TensorBoard.
    inputs : tensor
        The output of the immediately previous convolutional layer. 
    route : tensor
        The output of a previous convolutional layer.

    Returns
    -------
    inputs : tensor
        The resulting tensor after the concatenation operation between the 
        inputs tensor and the route tensor.
    """

    with tf.variable_scope(name):

        inputs = tf.concat([inputs, route], axis=-1)
    
    return inputs