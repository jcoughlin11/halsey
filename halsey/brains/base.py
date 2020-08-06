"""
Title: base.py
Notes:
"""
from abc import ABC
from abc import abstractmethod


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
    # build_networks
    # -----
    def build_networks(self, inputShape, nActions):
        """
        The neural networks cannot be built until the shape of the
        input data and the size of the game's action space are known.
        The size of the action space is used to determine the output
        shape. This method should be called by the instructor or
        proctor before work begins.
        """
        for i in range(len(self.nets)):
            self.nets[i].build_arch(inputShape, nActions)

    # -----
    # pre_populate
    # -----
    def pre_populate(self, navigator):
        """
        Fills the memory buffer before a run so that there are samples
        to draw from for learning early on.
        """
        self.memory.pre_populate(navigator)

    # -----
    # add_memory
    # -----
    def add_memory(self, experience):
        """
        Adds an experience to the memory buffer.
        """
        self.memory.add_memory(experience)

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
