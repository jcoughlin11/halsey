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
