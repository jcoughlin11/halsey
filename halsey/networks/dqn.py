"""
Title: dqn.py
Notes:
"""
import gin
import tensorflow as tf

from .base import BaseNetwork


# ============================================
#                    DQN
# ============================================
@gin.configurable
class DQN(BaseNetwork):
    """
    Constructs the original deep Q-learning neural network from [1]_.
    It consists of two convolutional layers followed by a
    fully-connected layer and then output layer. See the last
    paragraph of section 4.1 in [1]_.

    Attributes
    ----------
    pass

    Methods
    -------
    pass

    References
    ----------
    .. [1] Minh, V., **et al**., "Playing Atari with Deep
        Reinforcement Learning," CoRR, vol. 1312, 2013.
    """

    # -----
    # constructor
    # -----
    def __init__(self, inputShape, nActions, channelsFirst, params):
        """
        Defines the layers of the network.

        Parameters
        ----------
        pass

        Raises
        ------
        pass

        Returns
        -------
        pass
        """
        super().__init__(channelsFirst, params)
        # First convolutional layer
        self.conv1 = tf.keras.layers.Conv2D(
            input_shape=inputShape,
            data_format=self.dataFormat,
            filters=16,
            kernel_size=[8, 8],
            strides=[4, 4],
            activation="relu",
            name="conv1",
        )
        # Second convolutional layer
        self.conv2 = tf.keras.layers.Conv2D(
            data_format=self.dataFormat,
            filters=32,
            kernel_size=[4, 4],
            strides=[2, 2],
            activation="relu",
            name="conv2",
        )
        # Flatten layer
        self.flatten = tf.keras.layers.Flatten(data_format=self.dataFormat)
        # Fully connected layer
        self.d1 = tf.keras.layers.Dense(
            units=256, activation="relu", name="fc1"
        )
        # Output layer
        self.outputLayer = tf.keras.layers.Dense(
            units=nActions, activation="linear"
        )

    # -----
    # call
    # -----
    def call(self, x):
        """
        Defines a forward pass through the network.

        Parameters
        ----------
        x : tf.Variable
            The input state to the network.

        Raises
        ------
        pass

        Returns
        -------
        output : tf.Tensor
            The tensor containing the network's beliefs about the
            quality of each action for the given input state.
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.outputLayer(x)
