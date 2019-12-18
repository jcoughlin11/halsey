"""
Title: utils.py
Purpose:
Notes:
"""
import tensorflow as tf

from anna.utils import registers

from . import conv
from . import dueling
from . import rnn


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
    Doc string.

    Parameters:
    -----------
        pass

    Raises:
    -------
        pass

    Returns:
    --------
        pass
    """
    # Set up network architecture
    if arch in registers.convNets:
        net = conv.utils.build_net(arch, inputShape, channelsFirst, nActions)
    elif arch in registers.duelingNets:
        net = dueling.utils.build_net(arch, inputShape, channelsFirst, nActions)
    elif arch in registers.rnnNets:
        net = rnn.utils.build_net(arch, inputShape, channelsFirst, nActions)
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
    Doc string.

    Parameters:
    -----------
        pass

    Raises:
    -------
        pass

    Returns:
    --------
        pass
    """
    if optimizerName == "adam":
        optimizer = tf.keras.optimizers.Adam(lr=learningRate)
    return optimizer


# ============================================
#                set_loss
# ============================================
def set_loss(lossName):
    """
    Doc string.

    Parameters:
    -----------
        pass

    Raises:
    -------
        pass

    Returns:
    --------
        pass
    """
    if lossName == "mse":
        loss = tf.keras.losses.MeanSquaredError()
    return loss