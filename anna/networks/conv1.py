"""
Title:   conv1.py
Author:  Jared Coughlin
Date:    8/27/19
Purpose: Contains the conv1 architecture
Notes:
"""


#============================================
#             build_conv1_net
#============================================
def build_conv1_net(self):
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
    # Initialize empty model
    model = tf.keras.Sequential()
    # First convolutional layer
    model.add(
        tf.keras.layers.Conv2D(
            input_shape=self.inputShape,
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
    model.add(
        tf.keras.layers.Dense(units=256, activation="relu", name="fc1")
    )
    # Output layer
    model.add(
        tf.keras.layers.Dense(units=self.nActions, activation="linear")
    )
    return model
