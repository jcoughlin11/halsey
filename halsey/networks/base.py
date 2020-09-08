"""
Title: base.py
Notes:
"""
from abc import ABC
from abc import abstractmethod

import tensorflow as tf

from halsey.utils.gpu import using_gpu
from halsey.utils.register import register


# ============================================
#                 BaseNetwork
# ============================================
class BaseNetwork(ABC, tf.keras.Model):
    """
    The `network`(s) contain the weights. These weight parameters are
    tuned during the learning process in order to increase the agent's
    performance. The weights are also arranged into a specific
    architecture, which is also held by this object. Both the value and
    architecture of the weights influence performance.
    """
    networkType = None

    # -----
    # subclass hook
    # -----
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register(cls)

    # -----
    # constructor
    # -----
    def __init__(self, optimizer, lossObject, params):
        tf.keras.Model.__init__(self)
        self.params = params
        self.lossObject = lossObject
        self.optimizer = optimizer 

    # -----
    # build_arch
    # -----
    @abstractmethod
    def build_arch(self, inputShape, nActions):
        """
        Constructs the layers of the network.
        """
        pass

    # -----
    # call
    # -----
    @abstractmethod
    def call(self, x):
        """
        Defines a forward pass through the network.
        """
        pass

    # -----
    # get_loss_function
    # -----
    def get_loss_function(self, lossName):
        """
        Assigns the function object containing the implementation of
        the given loss function.
        """
        loss = tf.keras.losses.get(lossName)
        return loss

    # -----
    # get_optimizer
    # -----
    def get_optimizer(self, optimizerName, learningRate):
        """
        Assigns the class containing the implementation of the given
        optimizer.
        """
        optimizer = tf.keras.optimizers.get(optimizerName)
        optimizer.learning_rate = learningRate
        return optimizer
