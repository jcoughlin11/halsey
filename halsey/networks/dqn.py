"""
Title: dqn.py
Notes:
"""
import tensorflow as tf

from .base import BaseNetwork


# ============================================
#               DeepQNetwork
# ============================================
class DeepQNetwork(BaseNetwork):
    """
    The network architecture from Mnih et al. 2013.
    """
    networkType = "convolution"

    # -----
    # build_arch
    # -----
    def build_arch(self, inputShape, nLogits, dataFormat):
        """
        Constructs the layers of the network.
        """
        # First convolutional layer
        self.conv1 = tf.keras.layers.Conv2D(
            input_shape=inputShape,
            data_format=dataFormat,
            filters=16,
            kernel_size=[8, 8],
            strides=[4, 4],
            activation="relu",
            name="conv1",
        )
        # Second convolutional layer
        self.conv2 = tf.keras.layers.Conv2D(
            data_format=dataFormat,
            filters=32,
            kernel_size=[4, 4],
            strides=[2, 2],
            activation="relu",
            name="conv2",
        )
        # Flatten layer
        self.flatten = tf.keras.layers.Flatten(data_format=dataFormat)
        # Fully connected layer
        self.d1 = tf.keras.layers.Dense(
            units=256, activation="relu", name="d1"
        )
        # Output layer
        self.outputLayer = tf.keras.layers.Dense(
            units=nLogits, activation="linear", name="output"
        )

    # -----
    # call
    # -----
    def call(self, x):
        """
        Defines a forward pass through the network.
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.outputLayer(x)
