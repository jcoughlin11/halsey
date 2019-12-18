"""
Title: conv1.py
Purpose:
Notes:
"""
import tensorflow as tf


# ============================================
#              build_conv1_net
# ============================================
def build_conv1_net(inputShape, channelsFirst, nActions):
    """
    Constructs the layers of the network. Uses two convolutional
    layers followed by a FC and then output layer. See the last
    paragraph of section 4.1 in Mnih13.

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
