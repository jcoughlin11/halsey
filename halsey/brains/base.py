"""
Title: halsey.brains.base.py
Notes:
"""
from abc import ABC
from abc import abstractmethod

from halsey.utils.register import register


# ============================================
#                  BaseBrain
# ============================================
class BaseBrain(ABC):
    """
    Manages the memory, neural networks, and contains the learning
    method by which the neural network weights are updated (e.g., Q
    learning).
    """

    # -----
    # subclass hook
    # -----
    def __init_subclass__(cls, **kwargs):
        """
        Adds subclasses to the register used during set up.
        """
        super().__init_subclass__(**kwargs)
        register(cls)

    # -----
    # constructor
    # -----
    def __init__(self, memory, networks, params):
        self.memory = memory
        self.nets = networks
        self.params = params
        self.dataFormat = None
        self.inputShape = None
        self.nLogits = None

    # -----
    # build_networks
    # -----
    def build_networks(self):
        """
        The neural networks cannot be built until the shape of the
        input data and the size of the game's action space are known.
        The size of the action space is used to determine the output
        shape. This method should be called by the instructor or
        proctor before work begins.
        """
        for i in range(len(self.nets)):
            self.nets[i].build_arch(self.inputShape, self.nLogits, self.dataFormat)

    # -----
    # pre_populate
    # -----
    def pre_populate(self, navigator):
        """
        Fills the memory buffer before training so that there are
        samples to draw from for learning early on.
        """
        self.memory.pre_populate(navigator)

    # -----
    # add_memory
    # -----
    def add_memory(self, experience):
        """
        Puts an experience in the memory buffer.
        """
        self.memory.add_memory(experience)

    # -----
    # learn
    # -----
    @abstractmethod
    def learn(self):
        """
        The learning method by which the neural network weights are
        updated (e.g., Q-learning).
        """
        pass

    # -----
    # predict
    # -----
    @abstractmethod
    def predict(self, state):
        """
        Uses the neural network's current knowledge to select what it
        thinks is the best action for the current situation.
        """
        pass
