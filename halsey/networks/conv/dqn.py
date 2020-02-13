"""
Title: dqn.py
Purpose: Contains the conv1 neural network architecture.
Notes:
"""
import tensorflow as tf

from halsey.utils.validation import register_network


# ============================================
#                   Conv1
# ============================================
@register_network
class DQN(tf.keras.Model):
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

    networkType = "cnn"

    def __init__(self, inputShape, channelsFirst, nActions):
        """
        Defines the layers of the network.

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
        None
        """
        # Call Model's constructor
        super().__init__()
        # Set the data format
        if channelsFirst:
            dataFormat = "channels_first"
        else:
            dataFormat = "channels_last"
        # First convolutional layer
        self.conv1 = tf.keras.layers.Conv2D(
            input_shape=inputShape,
            data_format=dataFormat,
            filters=16,
            kernel_size=[8, 8],
            strides=[4, 4],
            name="conv1",
        )
        # Second convolutional layer
        self.conv2 = tf.keras.layers.Conv2D(
            data_format=dataFormat,
            filters=32,
            kernel_size=[4, 4],
            strides=[2, 2],
            name="conv2",
        )
        # Flatten layer
        self.flatten = tf.keras.layers.Flatten(data_format=dataFormat)
        # Fully connected layer
        self.d1 = tf.keras.layers.Dense(units=256, name="fc1")
        # Output layer
        self.outputLayer = tf.keras.layers.Dense(
            units=nActions, activation="linear"
        )

    def call(self, x):
        """
        Defines a forward pass through the network.

        .. warning::

            The activations have to be manually called. The layer version
            must also be used, otherwise an input node error is raised by
            tensorflow.

        Parameters
        ----------
        x : np.ndarray
            The input frame stack to the network.

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
        x = tf.keras.layers.ReLU()(x)
        x = self.conv2(x)
        x = tf.keras.layers.ReLU()(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = tf.keras.layers.ReLU()(x)
        return self.outputLayer(x)
