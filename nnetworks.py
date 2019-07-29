"""
Title:   nnetworks.py
Author:  Jared Coughlin
Date:    1/24/19
Purpose: Contains the neural network architecture definitions
Notes:
"""
import tensorflow as tf


#============================================
#            Deep-Q Network Class
#============================================
class DQN:
    """
    Defines the network architecture for Deep-Q learning.

    Attributes:
    -----------
        pass

    Methods:
    --------
        pass

    """

    # -----
    # Constructor
    # -----
    def __init__(self, architecture, inputShape, nActions, learningRate):
        """
        Parameters:
        -----------
            architecture : string
                The network architecture to use. These are specified as
                methods within this class, so this class can be used as
                a black box.

            inputShape : tuple
                This is the dimensions of the input image stack. It's
                batchSize x nrows x ncols x nstacked.

            nActions : int
                The size of the environment's action space

        Raises:
        -------
            pass

        Returns:
        --------
            pass
        """
        self.arch = architecture
        self.inputShape = inputShape
        self.nActions = nActions
        self.learningRate = learningRate
        # Build the network
        if self.arch == 'conv1':
            self.model = self.build_conv1_net()
        elif self.arch == 'dueling1':
            self.model = self.build_dueling1_net()
        elif self.arch == 'perdueling1':
            self.model = self.build_perdueling1_net()
        elif self.arch == 'rnn1':
            self.model = self.build_rnn1_net()
        else:
            raise ValueError("Error, unrecognized network architecture!")

    # -----
    # build_conv1_net
    # -----
    def build_conv1_net(self):
        """
        Constructs the layers of the network. Uses three convolutional
        layers followed by a FC and then output layer.

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
        # First convolutional layer. Keras sets padding to valid and
        # kernel_initializer to glorot_unitform (xavier)
        model.add(tf.keras.layers.Conv2D(
                input_shape=self.inputShape,
                filters=32,
                kernel_size=[8,8],
                strides=[4,4],
                activation='elu',
                name='conv1'
            )
        )
        # Second convolutional layer
        model.add(tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=[4,4],
                strides=[2,2],
                activation='elu',
                name='conv2'
            )
        )
        # Third convolutional layer
        model.add(tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=[3,3],
                strides=[2,2],
                activation='elu',
                name='conv3'
            )
        )
        # Flatten layer
        model.add(tf.keras.layers.Flatten())
        # Fully connected layer
        model.add(tf.keras.layers.Dense(
                units=512,
                activation='elu',
                name='fc1'
            )
        )
        # Output layer
        model.add(tf.keras.layers.Dense(units=self.nActions))
        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learningRate),
            loss='mse'
        )
        return model
