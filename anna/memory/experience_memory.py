"""
Title:   experience_memory.py
Purpose: Contains the object that holds and manages the vanilla
            Q-learning experience buffer.
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

    # -----
    # Pre-Populate
    # -----
    def pre_populate(self, navigator):
        """
        This function initially fills the experience buffer with sample
        experience tuples to avoid the empty memory problem.

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
        # Reset the environment
        navigator.reset()
        # Loop over the desired number of sample experiences
        for i in range(self.pretrainLen):
            experience = navigator.transition(mode="random")
            # Add experience to memory
            self.add(experience)

    # -----
    # Add
    # -----
    def add(self, experience):
        """
        Adds the newest experience tuple to the buffer.
        Parameters:
        -----------
            experience : tuple (or list of tuples in the case of an
                         RNN)
                Contains the state, action, reward, next_state, and done
                flag.
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
        This function returns a randomly selected subsample of size
        batch_size from the buffer. This subsample is used to train the
        DQN. Note that a deque's size is determined only from the
        elements that are in it, not from maxlen. That is, if you have a
        deque with maxlen = 10, but only one element has been added to
        it, then it's size is actually 1.

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
