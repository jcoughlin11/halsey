"""
Title: experiencememory.py
Purpose:
Notes:
"""
import collections

import numpy as np


# ============================================
#             ExperienceMemory
# ============================================
class ExperienceMemory:
    """
    Doc string.

    Attributes:
    -----------
        pass

    Methods:
    --------
        pass
    """

    # -----
    # constructor
    # -----
    def __init__(self, memoryParams):
        """
        Doc string.

        Parameters:
        -----------
            pass

        Raises:
        -------
            pass

        Returns:
        --------
            pass
        """
        self.maxSize = memoryParams.maxSize
        self.pretrainLen = memoryParams.pretrainLen
        self.buffer = collections.deque(maxlen=self.maxSize)
        self.isWeights = None

    # -----
    # Add
    # -----
    def add(self, experience):
        """
        Doc string.

        Parameters:
        -----------
            pass

        Raises:
        -------
            pass

        Returns:
        --------
            None
        """
        self.buffer.append(experience)

    # -----
    # Sample
    # -----
    def sample(self, batchSize):
        """
        Doc string.

        Parameters:
        -----------
            pass

        Raises:
        -------
            pass

        Returns:
        --------
            pass
        """
        # Choose random indices from the buffer. Make sure the
        # batch_size isn't larger than the current buffer size
        try:
            indices = np.random.choice(
                np.arange(len(self.buffer)), size=batchSize, replace=False
            )
        except ValueError:
            raise (
                "Error, need batch size < buf size when sampling from memory!"
            )
        # Select randomly chosen sample
        samples = np.array(self.buffer)[indices]
        # Set up arrays for holding parsed info. tf expects the batch
        # size to be the first argument of the shape
        states = np.empty(
            shape=[batchSize] + list(self.buffer[0].state.shape), dtype=np.float
        )
        actions = np.empty(shape=[batchSize, 1], dtype=np.int)
        rewards = np.empty(shape=[batchSize, 1], dtype=np.float)
        nextStates = np.empty(
            shape=[batchSize] + list(self.buffer[0].state.shape), dtype=np.float
        )
        dones = np.empty(shape=[batchSize, 1], dtype=np.bool)
        # Parse the sample
        for i, sample in enumerate(samples):
            states[i] = sample.state
            actions[i] = sample.action
            rewards[i] = sample.reward
            nextStates[i] = sample.nextState
            dones[i] = sample.done
        return states, actions, rewards, nextStates, dones
