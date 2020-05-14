"""
Title: base.py
Notes:
"""
import queue


# ============================================
#                BaseMemory
# ============================================
class BaseMemory:
    """
    Doc string.
    """

    # -----
    # constructor
    # -----
    def __init__(self, params):
        """
        Doc string.
        """
        self.maxSize = params["maxSize"]
        self.pretrainLen = params["pretrainLen"]
        self.replayBuffer = queue.deque(maxlen=self.maxSize)

    # -----
    # add
    # -----
    def add(self, experience):
        """
        Doc string.
        """
        self.replayBuffer.append(experience)
