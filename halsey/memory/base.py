"""
Title: base.py
Notes:
"""
import queue

import gin


# ============================================
#                BaseMemory
# ============================================
@gin.configurable("memory")
class BaseMemory:
    """
    Doc string.
    """

    # -----
    # constructor
    # -----
    def __init__(self, memoryParams):
        """
        Doc string.
        """
        self.maxSize = memoryParams["maxSize"]
        self.pretrainLen = memoryParams["pretrainLen"]
        self.replayBuffer = queue.deque(maxlen=self.maxSize)

    # -----
    # add
    # -----
    def add(self, experience):
        """
        Doc string.
        """
        self.replayBuffer.append(experience)

    # -----
    # pre_populate
    # -----
    def pre_populate(self, navigator):
        """
        Doc string.
        """
        navigator.reset()
        for _ in range(self.pretrainLen):
            experience = navigator.transition(None, "random")
            self.add(experience)
            # Check for terminal state
            if experience[-1]:
                navigator.reset()
