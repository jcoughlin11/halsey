"""
Title:   nnetworks.py
Author:  Jared Coughlin
Date:    1/24/19
Purpose: Contains the neural network architecture definitions
Notes:
    1.) For RNNs, see:
        https://www.tensorflow.org/beta/guide/keras/rnn
"""
import tensorflow as tf
import tensorflow.keras.backend as K


# ============================================
#            Deep-Q Network Class
# ============================================
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
    def __init__(
        self, architecture, inputShape, nActions, learningRate, optimizer, loss
    ):
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
        if self.arch == "conv1":
            self.model = self.build_conv1_net()
        elif self.arch == "dueling1":
            self.model = self.build_dueling1_net()
        elif self.arch == "rnn1":
            self.model = self.build_rnn1_net()
        else:
            raise ValueError("Error, unrecognized network architecture!")
        # Compile the model
        self.model.compile(optimizer=optimizer, loss=loss)

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

    # -----
    # build_dueling1_net
    # -----
    def build_dueling1_net(self):
        """
        Uses the keras functional API to build the dueling DQN given in
        Wang et al. 2016.

        The basic idea is that calculating a Q value provides an
        estimate how how good a certain action is in a given state.
        However, it provides no information whatsoever on whether or not
        it is desirable to be in that state in the first place.

        Dueling DQN solves this by separating Q = V + A, where V is the
        value stream and it estimates how desirable being in the
        current state is. A is the advantage stream, and it estimates
        the quality of each action for the state.

        Also see:
        https://keras.io/getting-started/functional-api-guide/

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
        # Input layer
        inputLayer = tf.keras.layers.Input(shape=self.inputShape, name="input")
        # First convolutional layer
        conv1Layer = tf.keras.layers.Conv2D(
            input_shape=self.inputShape,
            filters=16,
            kernel_size=[8, 8],
            strides=[4, 4],
            activation="relu",
            name="conv1",
        )(inputLayer)
        # Second convolutional layer
        conv2Layer = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=[4, 4],
            strides=[2, 2],
            activation="relu",
            name="conv2",
        )(conv1Layer)
        # Flatten layer
        flattenLayer = tf.keras.layers.Flatten()(conv2Layer)
        # Value stream
        valueFC1 = tf.keras.layers.Dense(
            units=512, activation="relu", name="value_fc1"
        )(flattenLayer)
        value = tf.keras.layers.Dense(units=1, activation=None, name="value")(
            valueFC1
        )
        # Advantage stream
        advantageFC1 = tf.keras.layers.Dense(
            units=512, activation="relu", name="activation_fc1"
        )(flattenLayer)
        advantage = tf.keras.layers.Dense(
            units=self.nActions, activation=None, name="advantage"
        )(advantageFC1)
        # Aggregation layer: (eq. 9 in paper)
        # Q(s,a) = V(s) + [A(s,a) - 1/|A| * sum_{a'} A(s,a')]
        # Using tf ops here results in an error when saving:
        # https://tinyurl.com/y5hyn8zh
        Q = tf.keras.layers.Lambda(
            lambda q: q[0] + (q[1] - K.mean(q[1], axis=1, keepdims=True))
        )([value, advantage])
        # Set the model
        model = tf.keras.models.Model(inputs=inputLayer, outputs=Q)
        return model

    #-----
    # build_rnn1_net
    #-----
    def build_rnn1_net(self):
        """
        Architecture for Recurrent Deep Q-Learning, from Hausknecht et
        al. 2017.

        See: https://www.tensorflow.org/beta/guide/keras/rnn
        and: https://tinyurl.com/y5g4f9hn (input and output shape for
            LSTMs)
        and: https://tinyurl.com/y28llddb (CNN LSTM,
            https://arxiv.org/abs/1411.4389)
        and: https://tinyurl.com/y28tkxak (how to use TimeDistributed)

        The input shape to an LSTM needs to be:
        (batchSize, nTimeSteps, nFeatures). A CNN, though doesn't care
        about time, just space, so it's shape is:
        (batchSize, nFeaturesDim1, ...)

        The way that the convolutional layers are applied to every
        time-step is to use the TimeDistributed layer. We pass the
        normal spatial shape to the CNN layer and then pass nTimeSteps
        to the wrapping TimeDistributed layer. All of this is then
        passed normally to the LSTM layer(s).

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
        model = tf.keras.models.Sequential()
        # First convolutional layer. Here, inputShape should be
        # (nTimeSteps, nFeaturesDim1, ..., nFeaturesDimN). The batch
        # size is handled when calling fit(). So, the true shape of the
        # input passed to fit is (batchSize, nTimeSteps,
        # nFeaturesDim1,...)
        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2D(
                    filters=32,
                    kernel_size=[8,8],
                    strides=[4,4],
                    activation='relu',
                    name='conv1'
                ),
                input_shape=self.inputShape
            )
        )
        # Second convolutional layer
        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=[4,4],
                    strides=[2,2],
                    activation='relu',
                    name='conv2'
                )
            )
        )
        # Third convolutional layer
        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=[3,3],
                    strides=[1,1],
                    activation='relu',
                    name='conv3'
                )
            )
        )
        # Flatten layer
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()))
        # LSTM
        model.add(
            tf.keras.layers.LSTM(
                units=512,
                name='lstm1'
            )
        )
        # FC layer
        model.add(
            tf.keras.layers.Dense(
                units=self.nActions,
                activation='linear',
                name='output'
            )
        )
        return model
