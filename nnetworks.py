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
    def __init__(self, architecture, inputShape, nActions, learningRate,
        optimizer, loss):
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
        self.optimizer = optimizer
        self.loss = loss
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
        # Compile the model
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
        )

    # -----
    # build_conv1_net
    # -----
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
        model.add(tf.keras.layers.Conv2D(
                input_shape=self.inputShape,
                filters=16,
                kernel_size=[8,8],
                strides=[4,4],
                activation='relu',
                name='conv1'
            )
        )
        # Second convolutional layer
        model.add(tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=[4,4],
                strides=[2,2],
                activation='relu',
                name='conv2'
            )
        )
        # Flatten layer
        model.add(tf.keras.layers.Flatten())
        # Fully connected layer
        model.add(tf.keras.layers.Dense(
                units=256,
                activation='relu',
                name='fc1'
            )
        )
        # Output layer
        model.add(tf.keras.layers.Dense(
            units=self.nActions,
            activation='linear'
            )
        )
        return model
