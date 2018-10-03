import tensorflow as tf

def shortcut_layer(name: str, shortcut, inputs):
    """
    Creates the typical residual block architecture. Residual blocks are useful 
    for training very deep convolutional neural networks because they act as
    gradient 'highways' that enable the gradient to flow back into the first few 
    initial convolutional layers. Without residual blocks, the gradient tends to
    disappear at those first inital layers and the model has a difficult time
    converging. 

    Parameters
    ----------
    name : string
        The name of the tensor to be used in TensorBoard.
    shortcut: tensor
        The output of a previous convolutional layer
    inputs : tensor
        The output of the immediately previous convolutional layer. 

    Returns
    -------
    inputs : tensor
        The resulting tensor.
    new_shortcut : tensor
        A new shortcut for a future residual block to connect to.
    """

    with tf.variable_scope(name):

        inputs += shortcut
        new_shortcut = inputs

    return inputs, new_shortcut