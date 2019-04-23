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



#============================================
#            Deep-Q Network Class
#============================================
class DQN():
    """
    This is the Deep-Q Network class. It defines the network architecture for training the
    agent using Deep-Q learning.

    Parameters:
    -----------
        name : string
            The name that tensorflow assigns to the variable namespace

        architecture : string
            The network architecture to use. These are specified as methods within this
            class, so this class can be used as a black box.

        inputShape : tuple
            This is the dimensions of the input image stack. It's
            batchSize x nrows x ncols x nstacked.

        nActions : int
            The size of the environment's action space
    """
    #-----
    # Constructor
    #-----
    def __init__(self, netName, architecture, inputShape, nActions, learningRate):
        self.name  = netName
        self.arch  = architecture
        self.inputShape = inputShape
        self.nActions = nActions
        self.learningRate = learningRate
        # Build the network
        if self.arch == 'conv1':
            self.build_conv1_net()
        elif self.arch == 'dueling1':
            self.build_dueling1_net()
        elif self.arch == 'perdueling1'
            self.build_perdueling1_net()
        else:
            raise ValueError("Error, unrecognized network architecture!")

    #-----
    # build_conv1_net
    #-----
    def build_conv1_net(self):
        """
        This function constructs the layers of the network. Uses three convolutional
        layers followed by a FC and then output layer
        """
        with tf.variable_scope(self.name):
            # Placeholders (anything that needs to be fed from the outside)
            # Input. This is the stack of frames from the game
            self.inputs = tf.placeholder(tf.float32,
                                        shape=self.inputShape,
                                        name='inputs')
            # Actions. The action the agent chose. Used to get the predicted Q value
            self.actions = tf.placeholder(tf.float32,
                                          shape=[None, 1],
                                          name='actions')
            # Target Q. The max discounted future reward playing from next state
            # after taking chosen action. Determined by Bellmann equation.
            self.target_Q = tf.placeholder(tf.float32,
                                           shape=[None],
                                           name='target')
            # First convolutional layer
            self.conv1 = tf.layers.conv2d(inputs=self.inputs,
                filters=32,
                kernel_size=[8,8],
                strides=[4,4],
                padding='VALID',
                kernel_initializer=xavier_initializer_conv2d(),
                name='conv1')
            # First convolutional layer activation
            self.conv1_out = tf.nn.elu(self.conv1, name='conv1_out')
            # Second convolutional layer
            self.conv2 = tf.layers.conv2d(inputs=self.conv1_out,
                filters=64,
                kernel_size=[4,4],
                strides=[2,2],
                padding='VALID',
                kernel_initializer=xavier_initializer_conv2d(),
                name='conv2')
            # Second convolutional layer activation
            self.conv2_out = tf.nn.elu(self.conv2, name='conv2_out')
            # Third convolutional layer
            self.conv3 = tf.layers.conv2d(inputs=self.conv2_out,
                filters=64,
                kernel_size=[3,3],
                strides=[2,2],
                padding='VALID',
                kernel_initializer=tf.xavier_initializer_conv2d(),
                name='conv3')
            # Third convolutional layer activation
            self.conv3_out = tf.nn.elu(self.conv3, name='conv3_out')
            # Flatten
            self.flatten = tf.contrib.layers.flatten(self.conv3_out)
            # FC layer
            self.fc = tf.layers.dense(inputs=self.flatten,
                units=512,
                activation=tf.nn.elu,
                kernel_initializer=tf.xavier_initializer(),
                name='fc1')
            # Output layer (FC)
            self.output = tf.layers.dense(inputs=self.fc,
                units=self.nActions,
                activation=None,
                kernel_initializer=xavier_initializer())
            # Get the predicted Q value. 
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions))
            # Get the error (MSE)
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
            # Optimizer
            self.optimizer = AdamOptimizer(self.learningRate).minimize(self.loss)
        self.saver = tf.train.Saver()

    #-----
    # build_dueling1_net
    #-----
    def build_dueling1_net(self):
        """
        This function constructs the layers of the network. Three conv layers followed by
        dueling DQN, which splits into two streams: one for value and one for advantage
        before recominging them with an aggregation layer
        """
        with tf.variable_scope(self.name):
            # Placeholders (anything that needs to be fed from the outside)
            # Input. This is the stack of frames from the game
            self.inputs = tf.placeholder(tf.float32,
                                        shape=self.inputShape,
                                        name='inputs')
            # Actions. The action the agent chose. Used to get the predicted Q value
            self.actions = tf.placeholder(tf.float32,
                                          shape=[None, 1],
                                          name='actions')
            # Target Q. The max discounted future reward playing from next state
            # after taking chosen action. Determined by Bellmann equation.
            self.target_Q = tf.placeholder(tf.float32,
                                           shape=[None],
                                           name='target')
            # First convolutional layer
            self.conv1 = tf.layers.conv2d(inputs=self.inputs,
                filters=32,
                kernel_size=[8,8],
                strides=[4,4],
                padding='VALID',
                kernel_initializer=xavier_initializer_conv2d(),
                name='conv1')
            # First convolutional layer activation
            self.conv1_out = tf.nn.elu(self.conv1, name='conv1_out')
            # Second convolutional layer
            self.conv2 = tf.layers.conv2d(inputs=self.conv1_out,
                filters=64,
                kernel_size=[4,4],
                strides=[2,2],
                padding='VALID',
                kernel_initializer=xavier_initializer_conv2d(),
                name='conv2')
            # Second convolutional layer activation
            self.conv2_out = tf.nn.elu(self.conv2, name='conv2_out')
            # Third convolutional layer
            self.conv3 = tf.layers.conv2d(inputs=self.conv2_out,
                filters=64,
                kernel_size=[3,3],
                strides=[2,2],
                padding='VALID',
                kernel_initializer=xavier_initializer_conv2d(),
                name='conv3')
            # Third convolutional layer activation
            self.conv3_out = tf.nn.elu(self.conv3, name='conv3_out')
            # Flatten
            self.flatten = tf.contrib.layers.flatten(self.conv3_out)
            # Value stream
            self.value_fc = tf.layers.dense(inputs=self.flatten,
                                            units=512,
                                            activation=tf.nn.elu,
                                            kernel_initializer=xavier_initializer(),
                                            name='value_fc')
            self.value = tf.layers.dense(inputs=self.value_fc,
                                        units=1,
                                        activation=None,
                                        kernel_initializer=xavier_initializer(),
                                        name='value')
            # Advantage stream
            self.advantage_fc = tf.layers.dense(inputs=self.flatten,
                                                units=512,
                                                activation=tf.nn.elu,
                                                kernel_initializer=xavier_initializer(),
                                                name='advantage_fc')
            self.advantage = tf.layers.dense(inputs=self.advantage_fc,
                                            units=self.nActions,
                                            activation=None,
                                            kernel_initializer=xavier_initializer(),
                                            name='advantage')
            # Aggregation layer: Q(s,a) = V(s) + [A(s,a) - 1/|A| * sum_{a'} A(s,a')]
            self.output = self.value + tf.subtract(self.advantage,
                            tf.reduce_mean(self.advantage, axis=1, keepdims=True))
            # Predicted Q value
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions), axis=1)
            # Loss: mse between predicted and target Q values
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
            self.optimizer = RMSPropOptimizer(self.learningRate).minimize(self.loss)
        self.saver = tf.train.Saver()

    #-----
    # build_perdueling1_net
    #-----
    def build_perdueling1_net(self):
        """
        This function is identical to dueling1, except that it has extra parameters in
        order to facilitate PER (prioritized experience replay)
        """
        with tf.variable_scope(self.name):
            # Placeholders
            self.inputs = tf.placeholder(tf.float32,
                                        shape=self.intputShape,
                                        name="inputs")
            self.isWeights = tf.placeholder(tf.float32,
                                            shape=[None,1],
                                            name='isWeights')
            self.actions = tf.placeholder(tf.float32,
                                        shape=[None, 1],
                                        name="actions")
            self.target_Q = tf.placeholder(tf.float32,
                                        shape=[None],
                                        name="target")
            # First convolutional layer
            self.conv1 = tf.layers.conv2d(inputs=self.inputs,
                filters=32,
                kernel_size=[8,8],
                strides=[4,4],
                padding='VALID',
                kernel_initializer=xavier_initializer_conv2d(),
                name='conv1')
            # First convolutional layer activation
            self.conv1_out = tf.nn.elu(self.conv1, name='conv1_out')
            # Second convolutional layer
            self.conv2 = tf.layers.conv2d(inputs=self.conv1_out,
                filters=64,
                kernel_size=[4,4],
                strides=[2,2],
                padding='VALID',
                kernel_initializer=xavier_initializer_conv2d(),
                name='conv2')
            # Second convolutional layer activation
            self.conv2_out = tf.nn.elu(self.conv2, name='conv2_out')
            # Third convolutional layer
            self.conv3 = tf.layers.conv2d(inputs=self.conv2_out,
                filters=64,
                kernel_size=[3,3],
                strides=[2,2],
                padding='VALID',
                kernel_initializer=xavier_initializer_conv2d(),
                name='conv3')
            # Third convolutional layer activation
            self.conv3_out = tf.nn.elu(self.conv3, name='conv3_out')
            # Flatten
            self.flatten = tf.contrib.layers.flatten(self.conv3_out)
            # Value stream
            self.value_fc = tf.layers.dense(inputs=self.flatten,
                                            units=512,
                                            activation=tf.nn.elu,
                                            kernel_initializer=xavier_initializer(),
                                            name='value_fc')
            self.value = tf.layers.dense(inputs=self.value_fc,
                                        units=1,
                                        activation=None,
                                        kernel_initializer=xavier_initializer(),
                                        name='value')
            # Advantage stream
            self.advantage_fc = tf.layers.dense(inputs=self.flatten,
                                                units=512,
                                                activation=tf.nn.elu,
                                                kernel_initializer=xavier_initializer(),
                                                name='advantage_fc')
            self.advantage = tf.layers.dense(inputs=self.advantage_fc,
                                            units=self.nActions,
                                            activation=None,
                                            kernel_initializer=xavier_initializer(),
                                            name='advantage')
            # Aggregation layer: Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))
            self.output = self.value + tf.subtract(self.advantage,
                          tf.reduce_mean(self.advantage, axis=1, keepdims=True))
            # Predicted Q value.
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)
            # Get absolute errors, which are used to update the sum tree
            self.absErrors = tf.abs(self.target_Q - self.Q)
            # The mse loss is modified because of PER 
            self.loss = tf.reduce_mean(self.isWeights *
                                        tf.squared_difference(self.target_Q, self.Q))
            self.optimizer = RMSPropOptimizer(self.learning_rate).minimize(self.loss)
        self.saver = tf.train.Saver()
