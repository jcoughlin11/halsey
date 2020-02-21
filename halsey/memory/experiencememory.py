"""
Title:      experiencememory.py
Purpose:    Contains the ExperienceMemory class.
Notes:
"""
import queue

import gin
import numpy as np

from halsey.utils.endrun import endrun

from .base import BaseMemory


# ============================================
#              ExperienceMemory
# ============================================
@gin.configurable
class ExperienceMemory(BaseMemory):
    """
    This is an object for storing and managing individual experiences.

    Attributes
    ----------
    buffer : collections.deque
        The container for the experiences.

    Methods
    -------
    sample(batchSize)
        Extracts a sample of batchSize experiences from the buffer to
        be used in training.

    See Also
    --------
    :py:class:`~halsey.memory.basebrain.BaseBrain`

    """

    # -----
    # constructor
    # -----
    def __init__(self, memoryParams):
        """
        Sets up the buffer.

        Parameters
        ----------
        memmoryParams : dict
            Contains the memory-specific data read from the gin config
            file.

        Raises
        ------
        None

        Returns
        -------
        None
        """
        super().__init__(memoryParams)
        self.buffer = queue.deque(maxlen=self.maxSize)

    # -----
    # Sample
    # -----
    def sample(self):
        """
        Extracts batchSize experiences from the buffer, processes them
        into the form required by the network, and returns the sample.

        Currently, this method selects each experience randomly, with
        each experience having an equal chance to be chosen.
        Prioritized Experience Replay (PER) will be added in the
        future.

        Parameters
        ----------
        None

        Raises
        ------
        None

        Returns
        -------
        states : np.ndarray
            Array containing the state encountered from each chosen
            experience.

        actions : np.ndarray
            Array containing the actions chosen for each encountered
            state.

        rewards : np.ndarray
            Array of the rewards given by the game for having taken the
            selected action in the encountered state.

        nextStates : np.ndarray
            Array containing the states that the game transitioned to
            after the agent executed it's chosen action in the
            encountered state.

        dones : np.ndarray
            Array indicating whether or not each of the nextStates is a
            terminal state or not.

        """
        # Choose random indices from the buffer. Make sure the
        # batch_size isn't larger than the current buffer size
        try:
            indices = np.random.choice(
                np.arange(len(self.buffer)), size=self.batchSize, replace=False
            )
        except ValueError as e:
            msg = f"Batch size `{self.batchSize}` < replay buffer size "
            msg += f"`{self.maxSize}`"
            endrun(e, msg)
        # Select randomly chosen sample
        samples = np.array(self.buffer)[indices]
        # Set up arrays for holding parsed info. tf expects the batch
        # size to be the first argument of the shape
        states = np.empty(
            shape=[self.batchSize] + list(self.buffer[0].state.shape),
            dtype=np.float,
        )
        actions = np.empty(shape=[self.batchSize, 1], dtype=np.int)
        rewards = np.empty(shape=[self.batchSize, 1], dtype=np.float)
        nextStates = np.empty(
            shape=[self.batchSize] + list(self.buffer[0].state.shape),
            dtype=np.float,
        )
        dones = np.empty(shape=[self.batchSize, 1], dtype=np.bool)
        # Parse the sample
        for i, sample in enumerate(samples):
            states[i] = sample.state
            actions[i] = sample.action
            rewards[i] = sample.reward
            nextStates[i] = sample.nextState
            dones[i] = sample.done
        return states, actions, rewards, nextStates, dones
