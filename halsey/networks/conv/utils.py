"""
Title: utils.py
Purpose:
Notes:
"""
from . import conv1


# ============================================
#                   build_net
# ============================================
def build_net(arch, inputShape, channelsFirst, nActions):
    """
    Handles construction of a new neural network.

    Parameters
    ----------
    arch : str
        The name of the neural network architecture to use.

    inputShape : list
        The dimensions of the neural network's input. Is either NCHW
        (GPU or RNN) or NHWC (CPU).

    channelsFirst : bool
        If True, input shape is NCHW. If False, input shape is NHWC.

    nActions : int
        The size of the game's action space. Is used for determining
        the neural network's output shape.

    Raises
    ------
    None

    Returns
    -------
    net : tf.keras.Model
        The uncompiled neural network.
    """
    # Set up network architecture
    if arch == "conv1":
        net = conv1.build_conv1_net(inputShape, channelsFirst, nActions)
    return net
