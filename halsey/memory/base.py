"""
Title: base.py
Notes:
"""
from abc import ABC, abstractmethod

from halsey.utils.register import register


# ============================================
#                 BaseMemory
# ============================================
class BaseMemory(ABC):
    """
    The `memory` object is responsible for storing the experiences
    provided by the game in response to the agent's actions. It is
    also responsible for sampling from this buffer for learning.
    """

    # -----
    # subclass hook
    # -----
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register(cls)

    # -----
    # constructor
    # -----
    def __init__(self, params):
        self.params = params

    # -----
    # add_memory
    # -----
    @abstractmethod
    def add_memory(self, experience):
        """
        Adds the given experience to the memory buffer.
        """
        pass

    # -----
    # pre_populate
    # -----
    @abstractmethod
    def pre_populate(self, navigator):
        """
        Fills the initially empty memory buffer with experiences.
        """
        pass

    # -----
    # sample
    # -----
    @abstractmethod
    def sample(self):
        """
        Draws a batch of experiences from the memory buffer to be used
        in learning.
        """
        pass
