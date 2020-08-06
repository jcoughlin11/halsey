"""
Title: qmemory.py
Notes:
"""
import queue

from .base import BaseMemory


# ============================================
#                  QMemory
# ============================================
class QMemory(BaseMemory):
    """
    Standard replay buffer from Mnih et al. 2013.
    """

    # -----
    # constructor
    # -----
    def __init__(self, params):
        super().__init__(params)
        self.memoryBuffer = queue.deque(maxlen=self.params["maxSize"])

    # -----
    # add
    # -----
    def add(self, experience):
        """
        Adds the given experience to the memory buffer.
        """
        self.memoryBuffer.append(experience)

    # -----
    # pre_populate
    # -----
    def pre_populate(self, navigator):
        """
        Fills the initially empty memory buffer with experiences.
        """
        navigator.reset()
        for _ in range(self.params["pretrainLen"]):
            experience = navigator.transition(None, "random")
            self.add(experience)
            # Check for terminal state
            if experience[-1]:
                navigator.reset()

    # -----
    # sample
    # -----
    def sample(self):
        """
        Draws a batch of experiences from the memory buffer to be used
        in learning.
        """
