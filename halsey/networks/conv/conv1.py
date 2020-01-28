"""
Title: conv1.py
Purpose: Contains the conv1 neural network architecture.
Notes:
"""
import tensorflow as tf

import halsey


# ============================================
#              build_conv1_net
# ============================================
@halsey.utils.validation.register_network("conv1", "cnn")
def conv1(inputShape, channelsFirst, nActions):
    """
    Constructs the original deep Q-learning neural network from [1]_.
    It consists of two convolutional layers followed by a
    fully-connected layer and then output layer. See the last
    paragraph of section 4.1 in [1]_.

    Parameters
    ----------
    inputShape : list
        Contains the dimensions of the neural network's inputs. These
        are either NCHW (for GPU or RNN) or NHWC (for CPU).

    channelsFirst : bool
        If True, the input shape is NCHW. If False, the input shape is
        NHWC.

    nActions : int
        The size of the game's action space. Used to determine the
        shape of the neural network's output.

    Raises
    ------
    None

    Returns
    -------
    model : tf.keras.Model
        The constructed (but uncompiled) neural network).

    References
    ----------
    .. [1] Minh, V., **et al**., "Playing Atari with Deep
        Reinforcement Learning," CoRR, vol. 1312, 2013.
    """
    # Set the data format
    if channelsFirst:
        dataFormat = "channels_first"
    else:
        dataFormat = "channels_last"
    # Initialize empty model
    model = tf.keras.Sequential()
    # First convolutional layer
    model.add(
        tf.keras.layers.Conv2D(
            input_shape=inputShape,
            data_format=dataFormat,
            filters=16,
            kernel_size=[8, 8],
            strides=[4, 4],
            activation="relu",
            name="conv1",
        )
    )
    # Second convolutional layer
    model.add(
        tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=[4, 4],
            strides=[2, 2],
            activation="relu",
            name="conv2",
        )
    )
    # Flatten layer
    model.add(tf.keras.layers.Flatten())
    # Fully connected layer
    model.add(tf.keras.layers.Dense(units=256, activation="relu", name="fc1"))
    # Output layer
    model.add(tf.keras.layers.Dense(units=nActions, activation="linear"))
    return model
