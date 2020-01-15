"""
Title: utils.py
Purpose: Handles the construction of a new neural network.
Notes:
"""
import tensorflow as tf

from anna.utils.validation import registers

from . import conv


# ============================================
#               build_network
# ============================================
def build_network(
    arch,
    inputShape,
    channelsFirst,
    nActions,
    optimizerName,
    lossName,
    learningRate,
):
    """
    Handles construction of a new neural network.

    Parameters
    ----------
    arch : str
        The name of the neural network architecture to use.

    inputShape : list
        The dimensions of the neural network's input. Must be either
        NCHW (GPU or RNN) or NHWC (CPU).

    channelsFirst : bool
        If True, then the input shape is NCHW, otherwise it's NHWC.

    nActions : int
        The size of the game's action space. This is used in
        determining the shape of the neural network's output.

    optimizerName : str
        The name of the optimizer to use.

    lossName : str
        The name of the loss function to minimize.

    learningRate : float
        The step size to use during back propagation.

    Raises
    ------
    None

    Returns
    -------
    net : anna.networks.NeuralNetwork
        The newly created neural network.
    """
    # Set up network architecture
    if arch in registers.convNetRegister:
        net = conv.utils.build_net(arch, inputShape, channelsFirst, nActions)
    # Set the optimizer and loss functions appropriately
    optimizer = set_optimizer(optimizerName, learningRate)
    loss = set_loss(lossName)
    # Compile model
    net.compile(optimizer=optimizer, loss=loss)
    return net


# ============================================
#             set_optimizer
# ============================================
def set_optimizer(optimizerName, learningRate):
    """
    Assigns the actual optimizer function based on the given string
    form.

    This way of doing it allows for not only native keras optimizers,
    but also custom, user-defined optimizers, as well. Both
    user-defined and native keras optimizers are handled in
    the same manner, which makes life simpler, if potentially more
    verbose than necessary in certain cases.

    Parameters
    ----------
    optimizerName : str
        The name of the optimizer to use.

    learningRate : float
        The step size to use during back propagation.

    Raises
    ------
    None

    Returns
    -------
    optimizer : tf.keras.optimizers.Optimizer
        The actual optimizer object to perform minimization of the loss
        function.
    """
    if optimizerName == "adam":
        optimizer = tf.keras.optimizers.Adam(lr=learningRate)
    return optimizer


# ============================================
#                set_loss
# ============================================
def set_loss(lossName):
    """
    Sets the actual loss function to be minimized based on the given
    string form.

    This way of doing it allows for not only native keras losses,
    but also custom, user-defined losses, as well. Both
    user-defined and native keras lossesare handled in
    the same manner, which makes life simpler, if potentially more
    verbose than necessary in certain cases.

    Parameters
    ----------
    lossName : str
        The name of the loss function to use.

    Raises
    ------
    None

    Returns
    -------
    loss : tf.keras.losses.Loss
        The actual loss function to be minimized during training.
    """
    if lossName == "mse":
        loss = tf.keras.losses.MeanSquaredError()
    return loss
