"""
Title: experiencememory.py
Notes:
"""
import gin
import numpy as np

from halsey.utils.endrun import endrun
from halsey.utils.setup import prep_sample

from .base import BaseMemory


# ============================================
#            ExperienceMemory
# ============================================
@gin.configurable(whitelist=["memoryParams"])
class ExperienceMemory(BaseMemory):
    """
    Doc string.
    """

    # -----
    # constructor
    # -----
    def __init__(self, pipeline, memoryParams):
        """
        Doc string.
        """
        super().__init__(pipeline, memoryParams)

    # -----
    # sample
    # -----
    def sample(self, batchSize):
        """
        Doc string.
        """
        # Make sure batch size isn't larger than the replay buffer
        try:
            indices = np.random.choice(
                np.arange(len(self.replayBuffer)), size=batchSize, replace=False
            )
        except ValueError as e:
            msg = f"Batch size `{batchSize}` > buffer size "
            msg += f"`{len(self.replayBuffer)}`. Cannot sample."
            endrun(e, msg)
        sample = np.array(self.replayBuffer)[indices]
        sample = prep_sample(sample)
        return sample
