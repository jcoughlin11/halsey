"""
Title: base.py
Notes:
"""
from abc import ABC
from abc import abstractbaseclass


# ============================================
#                  BaseBrain
# ============================================
class BaseBrain(ABC):
    """
    The brain manages the memory and neural networks and contains the
    learning method by which the neural network weights are updated.
    """

    # -----
    # constructor
    # -----
    def __init__(self, memory, networks, params):
        self.memory = memory
        self.nets = networks
        self.params = params

    # -----
    # pre_populate
    # -----
    @abstractmethod
    def pre_populate(self):
        """
        Fills the memory buffer before a run so that there are samples
        to draw from for learning early on.
        """
        pass

    # -----
    # add_memory
    # -----
    @abstractmethod
    def add_memory(self, experience):
        """
        Adds an experience to the memory buffer.
        """
        pass

    # -----
    # learn
    # -----
    @abstractmethod
    def learn(self):
        """
        The learning method by which the neural network weights are
        updated.
        """
        pass

    # -----
    # predict
    # -----
    @abstractmethod
    def predict(self, state):
        """
        Uses the current knowledge of the neural network(s) to select
        what it thinks is the best action for the current situation.
        """
        pass
