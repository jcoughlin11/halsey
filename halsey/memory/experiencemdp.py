"""
Title: experiencemdp.py
Notes:
"""
import queue

import gin
import numpy as np

from halsey.utils.endrun import endrun


# ============================================
#              ExperienceMDP
# ============================================
@gin.configurable
class ExperienceMDPMemory:
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
        except ValueError:
            msg = f"Batch size `{batchSize}` > buffer size "
            msg += f"`{len(self.replayBuffer)}`. Cannot sample."
            endrun(msg)
        sample = np.array(self.replayBuffer)[indices]
        sample = self.prep_sample(sample)
        return sample

    # -----
    # pre_populate
    # -----
    def pre_populate(self, game):
        """
        Doc string.
        """
        game.reset()
        for _ in range(self.pretrainLen):
            experience = game.transition(mode="random")
            self.add(experience)
            # Check for terminal state
            if experience[-1]:
                game.reset()

    # -----
    # prep_sample
    # -----
    def prep_sample(self, sample):
        """
        Doc string.
        """
        states = np.stack(sample[:, 0]).astype(np.float)
        actions = sample[:, 1].astype(np.int)
        rewards = sample[:, 2].astype(np.float)
        nextStates = np.stack(sample[:, 3]).astype(np.float)
        dones = sample[:, 4].astype(np.bool)
        return (states, actions, rewards, nextStates, dones)
