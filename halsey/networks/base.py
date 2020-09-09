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
    def __init__(self, optimizer, loss, params):
        tf.keras.Model.__init__(self)
        self.params = params
        self.optimizer = optimizer 
        self.loss = loss

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
