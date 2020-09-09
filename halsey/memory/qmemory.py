"""
Title: qmemory.py
Notes:
"""
from queue import deque

import numpy as np

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
        self.memoryBuffer = deque(maxlen=self.params["maxSize"])

    # -----
    # add
    # -----
    def add_memory(self, experience):
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
            self.add_memory(experience)
            # Check for terminal state
            if experience[-1]:
                navigator.reset()

    # -----
    # prep_sample
    # -----
    def prep_sample(sample):
        """
        Batches all aspects of the sample.
        """
        states = np.stack(sample[:, 0]).astype(np.float)
        actions = sample[:, 1].astype(np.int)
        rewards = sample[:, 2].astype(np.float)
        nextStates = np.stack(sample[:, 3]).astype(np.float)
        dones = sample[:, 4].astype(np.bool)
        return (states, actions, rewards, nextStates, dones)

    # -----
    # sample
    # -----
    def sample(self):
        """
        Draws a batch of experiences from the memory buffer to be used
        in learning.
        """
        indices = np.random.choice(
            np.arange(len(self.memoryBuffer)), size=self.params["batchSize"], replace=False
        )
        sample = np.array(self.memoryBuffer)[indices]
        states = np.stack(sample[:, 0]).astype(np.float)
        actions = sample[:, 1].astype(np.int)
        rewards = sample[:, 2].astype(np.float)
        nextStates = np.stack(sample[:, 3]).astype(np.float)
        dones = sample[:, 4].astype(np.bool)
        return (states, actions, rewards, nextStates, dones)
