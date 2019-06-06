"""
Title:   nnetworks.py
Author:  Jared Coughlin
Date:    1/24/19
Purpose: Contains the class definitions of neural networks
Notes:
"""
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.layers import xavier_initializer_conv2d
from tensorflow.train import AdamOptimizer
from tensorflow.train import RMSPropOptimizer


# ============================================
#            Deep-Q Network Class
# ============================================
class DQN:
    """
    Defines the network architecture for training the agent using Deep-Q
    learning.

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
        self, netName, architecture, inputShape, nActions, learningRate):
        """
        Parameters:
        -----------
            name : string
                The name that tensorflow assigns to the variable
                namespace.

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
        self.name = netName
        self.arch = architecture
        self.inputShape = inputShape
        self.nActions = nActions
        self.learningRate = learningRate
        # Build the network
        if self.arch == "conv1":
            self.build_conv1_net()
        elif self.arch == "dueling1":
            self.build_dueling1_net()
        elif self.arch == "perdueling1":
            self.build_perdueling1_net()
        elif self.arch == "rnn1":
            self.build_rnn1_net()
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
        with tf.variable_scope(self.name):
            # Placeholders (anything that needs to be fed from the
            # outside)
            # Input. This is the stack of frames from the game
            self.inputs = tf.placeholder(
                tf.float32, shape=self.inputShape, name="inputs"
            )
            # Actions. The action the agent chose. Used to get the
            # predicted Q value
            self.actions = tf.placeholder(
                tf.float32, shape=[None, 1], name="actions"
            )
            # Target Q. The max discounted future reward playing from
            # next state after taking chosen action. Determined by
            # Bellmann equation.
            self.target_Q = tf.placeholder(
                tf.float32, shape=[None], name="target"
            )
            # First convolutional layer
            self.conv1 = tf.layers.conv2d(
                inputs=self.inputs,
                filters=32,
                kernel_size=[8, 8],
                strides=[4, 4],
                padding="VALID",
                kernel_initializer=xavier_initializer_conv2d(),
                name="conv1",
            )
            # First convolutional layer activation
            self.conv1_out = tf.nn.elu(self.conv1, name="conv1_out")
            # Second convolutional layer
            self.conv2 = tf.layers.conv2d(
                inputs=self.conv1_out,
                filters=64,
                kernel_size=[4, 4],
                strides=[2, 2],
                padding="VALID",
                kernel_initializer=xavier_initializer_conv2d(),
                name="conv2",
            )
            # Second convolutional layer activation
            self.conv2_out = tf.nn.elu(self.conv2, name="conv2_out")
            # Third convolutional layer
            self.conv3 = tf.layers.conv2d(
                inputs=self.conv2_out,
                filters=64,
                kernel_size=[3, 3],
                strides=[2, 2],
                padding="VALID",
                kernel_initializer=xavier_initializer_conv2d(),
                name="conv3",
            )
            # Third convolutional layer activation
            self.conv3_out = tf.nn.elu(self.conv3, name="conv3_out")
            # Flatten
            self.flatten = tf.contrib.layers.flatten(self.conv3_out)
            # FC layer
            self.fc = tf.layers.dense(
                inputs=self.flatten,
                units=512,
                activation=tf.nn.elu,
                kernel_initializer=xavier_initializer(),
                name="fc1",
            )
            # Output layer (FC)
            self.output = tf.layers.dense(
                inputs=self.fc,
                units=self.nActions,
                activation=None,
                kernel_initializer=xavier_initializer(),
            )
            # Get the predicted Q value.
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions))
            # Get the error (MSE)
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
            # Optimizer
            self.optimizer = AdamOptimizer(self.learningRate).minimize(
                self.loss
            )
        self.saver = tf.train.Saver()

    # -----
    # build_dueling1_net
    # -----
    def build_dueling1_net(self):
        """
        Constructs the layers of the network. Three conv layers followed
        by dueling DQN, which splits into two streams: one for value and
        one for advantage before recominging them with an aggregation
        layer.
        """
        with tf.variable_scope(self.name):
            # Placeholders (anything that needs to be fed from the
            # outside)
            # Input. This is the stack of frames from the game
            self.inputs = tf.placeholder(
                tf.float32, shape=self.inputShape, name="inputs"
            )
            # Actions. The action the agent chose. Used to get the
            # predicted Q value
            self.actions = tf.placeholder(
                tf.float32, shape=[None, 1], name="actions"
            )
            # Target Q. The max discounted future reward playing from
            # next state after taking chosen action. Determined by
            # Bellmann equation
            self.target_Q = tf.placeholder(
                tf.float32, shape=[None], name="target"
            )
            # First convolutional layer
            self.conv1 = tf.layers.conv2d(
                inputs=self.inputs,
                filters=32,
                kernel_size=[8, 8],
                strides=[4, 4],
                padding="VALID",
                kernel_initializer=xavier_initializer_conv2d(),
                name="conv1",
            )
            # First convolutional layer activation
            self.conv1_out = tf.nn.elu(self.conv1, name="conv1_out")
            # Second convolutional layer
            self.conv2 = tf.layers.conv2d(
                inputs=self.conv1_out,
                filters=64,
                kernel_size=[4, 4],
                strides=[2, 2],
                padding="VALID",
                kernel_initializer=xavier_initializer_conv2d(),
                name="conv2",
            )
            # Second convolutional layer activation
            self.conv2_out = tf.nn.elu(self.conv2, name="conv2_out")
            # Third convolutional layer
            self.conv3 = tf.layers.conv2d(
                inputs=self.conv2_out,
                filters=64,
                kernel_size=[3, 3],
                strides=[2, 2],
                padding="VALID",
                kernel_initializer=xavier_initializer_conv2d(),
                name="conv3",
            )
            # Third convolutional layer activation
            self.conv3_out = tf.nn.elu(self.conv3, name="conv3_out")
            # Flatten
            self.flatten = tf.contrib.layers.flatten(self.conv3_out)
            # Value stream
            self.value_fc = tf.layers.dense(
                inputs=self.flatten,
                units=512,
                activation=tf.nn.elu,
                kernel_initializer=xavier_initializer(),
                name="value_fc",
            )
            self.value = tf.layers.dense(
                inputs=self.value_fc,
                units=1,
                activation=None,
                kernel_initializer=xavier_initializer(),
                name="value",
            )
            # Advantage stream
            self.advantage_fc = tf.layers.dense(
                inputs=self.flatten,
                units=512,
                activation=tf.nn.elu,
                kernel_initializer=xavier_initializer(),
                name="advantage_fc",
            )
            self.advantage = tf.layers.dense(
                inputs=self.advantage_fc,
                units=self.nActions,
                activation=None,
                kernel_initializer=xavier_initializer(),
                name="advantage",
            )
            # Aggregation layer: Q(s,a) = V(s) + [A(s,a) - 1/|A| *
            # sum_{a'} A(s,a')]
            self.output = self.value + tf.subtract(
                self.advantage,
                tf.reduce_mean(self.advantage, axis=1, keepdims=True),
            )
            # Predicted Q value
            self.Q = tf.reduce_sum(
                tf.multiply(self.output, self.actions), axis=1
            )
            # Loss: mse between predicted and target Q values
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
            self.optimizer = RMSPropOptimizer(self.learningRate).minimize(
                self.loss
            )
        self.saver = tf.train.Saver()

    # -----
    # build_perdueling1_net
    # -----
    def build_perdueling1_net(self):
        """
        This function is identical to dueling1, except that it has extra
        parameters in order to facilitate PER (prioritized experience
        replay).

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
        with tf.variable_scope(self.name):
            # Placeholders
            self.inputs = tf.placeholder(
                tf.float32, shape=self.inputShape, name="inputs"
            )
            self.isWeights = tf.placeholder(
                tf.float32, shape=[None, 1], name="isWeights"
            )
            self.actions = tf.placeholder(
                tf.float32, shape=[None, 1], name="actions"
            )
            self.target_Q = tf.placeholder(
                tf.float32, shape=[None], name="target"
            )
            # First convolutional layer
            self.conv1 = tf.layers.conv2d(
                inputs=self.inputs,
                filters=32,
                kernel_size=[8, 8],
                strides=[4, 4],
                padding="VALID",
                kernel_initializer=xavier_initializer_conv2d(),
                name="conv1",
            )
            # First convolutional layer activation
            self.conv1_out = tf.nn.elu(self.conv1, name="conv1_out")
            # Second convolutional layer
            self.conv2 = tf.layers.conv2d(
                inputs=self.conv1_out,
                filters=64,
                kernel_size=[4, 4],
                strides=[2, 2],
                padding="VALID",
                kernel_initializer=xavier_initializer_conv2d(),
                name="conv2",
            )
            # Second convolutional layer activation
            self.conv2_out = tf.nn.elu(self.conv2, name="conv2_out")
            # Third convolutional layer
            self.conv3 = tf.layers.conv2d(
                inputs=self.conv2_out,
                filters=64,
                kernel_size=[3, 3],
                strides=[2, 2],
                padding="VALID",
                kernel_initializer=xavier_initializer_conv2d(),
                name="conv3",
            )
            # Third convolutional layer activation
            self.conv3_out = tf.nn.elu(self.conv3, name="conv3_out")
            # Flatten
            self.flatten = tf.contrib.layers.flatten(self.conv3_out)
            # Value stream
            self.value_fc = tf.layers.dense(
                inputs=self.flatten,
                units=512,
                activation=tf.nn.elu,
                kernel_initializer=xavier_initializer(),
                name="value_fc",
            )
            self.value = tf.layers.dense(
                inputs=self.value_fc,
                units=1,
                activation=None,
                kernel_initializer=xavier_initializer(),
                name="value",
            )
            # Advantage stream
            self.advantage_fc = tf.layers.dense(
                inputs=self.flatten,
                units=512,
                activation=tf.nn.elu,
                kernel_initializer=xavier_initializer(),
                name="advantage_fc",
            )
            self.advantage = tf.layers.dense(
                inputs=self.advantage_fc,
                units=self.nActions,
                activation=None,
                kernel_initializer=xavier_initializer(),
                name="advantage",
            )
            # Aggregation layer: Q(s,a) = V(s) + (A(s,a) - 1/|A| *
            # sum A(s,a'))
            self.output = self.value + tf.subtract(
                self.advantage,
                tf.reduce_mean(self.advantage, axis=1, keepdims=True),
            )
            # Predicted Q value.
            self.Q = tf.reduce_sum(
                tf.multiply(self.output, self.actions), axis=1
            )
            # Get absolute errors, which are used to update the sum tree
            self.absErrors = tf.abs(self.target_Q - self.Q)
            # The mse loss is modified because of PER
            self.loss = tf.reduce_mean(
                self.isWeights * tf.squared_difference(self.target_Q, self.Q)
            )
            self.optimizer = RMSPropOptimizer(self.learningRate).minimize(
                self.loss
            )
        self.saver = tf.train.Saver()

    #-----
    # build_rnn1_net
    #-----
    def build_rnn1_net(self):
        """
        Builds the DQ network, but instead of a FC layer before the
        output layer, there's a LSTM layer. This allows for only one
        frame to be passed instead of a stack of four. See:
        https://tinyurl.com/y4l8mxsm, and:
        https://colah.github.io/posts/2015-08-Understanding-LSTMs/

        Parameters:
        -----------
            None

        Raises:
        -------
            None

        Returns:
        --------
            None
        """
        with tf.variable_scope(self.name):
            # Input
            self.inputs =  tf.placeholder(
                tf.float32, shape=self.inputShape, name="inputs")
            # Actions. The action the agent chose. Used to get the
            # predicted Q value
            self.actions = tf.placeholder(
                tf.float32, shape=[None, 1], name="actions"
            )
            # Target Q. The max discounted future reward playing from
            # next state after taking chosen action. Determined by
            # Bellmann equation
            self.target_Q = tf.placeholder(
                tf.float32, shape=[None], name="target"
            )
            # Batchsize for the rnn
            self.batchSize = tf.placeholder(
                tf.float32, shape=[], name="batchSize")
            # First convolutional layer
            self.conv1 = tf.layers.conv2d(
                inputs=self.inputs,
                filters=32,
                kernel_size=[8, 8],
                strides=[4, 4],
                padding="VALID",
                kernel_initializer=xavier_initializer_conv2d(),
                name="conv1",
            )
            # First convolutional layer activation
            self.conv1_out = tf.nn.elu(self.conv1, name="conv1_out")
            # Second convolutional layer
            self.conv2 = tf.layers.conv2d(
                inputs=self.conv1_out,
                filters=64,
                kernel_size=[4, 4],
                strides=[2, 2],
                padding="VALID",
                kernel_initializer=xavier_initializer_conv2d(),
                name="conv2",
            )
            # Second convolutional layer activation
            self.conv2_out = tf.nn.elu(self.conv2, name="conv2_out")
            # Third convolutional layer
            self.conv3 = tf.layers.conv2d(
                inputs=self.conv2_out,
                filters=64,
                kernel_size=[3, 3],
                strides=[2, 2],
                padding="VALID",
                kernel_initializer=xavier_initializer_conv2d(),
                name="conv3",
            )
            # Third convolutional layer activation
            self.conv3_out = tf.nn.elu(self.conv3, name="conv3_out")
            # Flatten
            self.flatten = tf.contrib.layers.flatten(self.conv3_out)
            # Create the cell for the recurrent layer. It has the same
            # number of units as the output of the last convolutional
            # layer (this is in keeping with Hausknecht17, who were
            # looking at the effects of only adding recurrence to DQN
            # and nothing else. Since the FC layer before the output
            # layer in DQN has the same number of units as the output of
            # the last conv layer then that's what they did for their
            # recurrent layer, as well.
            self.rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=512)
            # Create the initial rnn state
            self.rnnInitState = self.rnn_cell.zero_state(
                self.BatchSize, tf.float32
            )
            # Recurrent network
            self.rnn, self.rnn_state = tf.nn.dynamic_rnn(
                inputs=self.flatten,
                cell=self.rnn_cell,
                dtype=tf.float32,
                initial_state=self.rnnInitState
            )
            # Reshape rnn to work with FC layers. Recall the a -1 in a
            # shape means to set the value for that dimension
            # automatically such that the total size of the tensor is
            # retained. It's used when the size of one of the dims is
            # not known. At most one dimension can have -1 as its shape
            # value
            self.rnn = tf.reshape(self.rnn, shape=[-1, 512])
            # Split the output of the rnn into two streams
            # Value stream
            self.value_fc = tf.layers.dense(
                inputs=self.rnn,
                units=512,
                activation=tf.nn.elu,
                kernel_initializer=xavier_initializer(),
                name="value_fc",
            )
            self.value = tf.layers.dense(
                inputs=self.value_fc,
                units=1,
                activation=None,
                kernel_initializer=xavier_initializer(),
                name="value",
            )
            # Advantage stream
            self.advantage_fc = tf.layers.dense(
                inputs=self.rnn,
                units=512,
                activation=tf.nn.elu,
                kernel_initializer=xavier_initializer(),
                name="advantage_fc",
            )
            self.advantage = tf.layers.dense(
                inputs=self.advantage_fc,
                units=self.nActions,
                activation=None,
                kernel_initializer=xavier_initializer(),
                name="advantage",
            )
            # Aggregation layer: Q(s,a) = V(s) + [A(s,a) - 1/|A| *
            # sum_{a'} A(s,a')]
            self.output = self.value + tf.subtract(
                self.advantage,
                tf.reduce_mean(self.advantage, axis=1, keepdims=True),
            )
            # Predicted Q value
            self.Q = tf.reduce_sum(
                tf.multiply(self.output, self.actions), axis=1
            )
            # Loss: mse between predicted and target Q values
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
            self.optimizer = RMSPropOptimizer(self.learningRate).minimize(
                self.loss
            )
        self.saver = tf.train.Saver()
