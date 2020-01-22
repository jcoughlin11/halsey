"""
Title: experiencememory.py
Purpose: Contains the experience memory class.
Notes:
"""
import collections

import numpy as np


# ============================================
#              ExperienceMemory
# ============================================
class ExperienceMemory:
    """
    This is an object for storing and managing individual experiences.

    Attributes
    ----------
    buffer : collections.deque
        The container for the experiences.

    isWeights : np.ndarray
        Currently unused, but are intended for use with prioritized
        experience replay (PER).

    maxSize : int
        The largest number of experiences to store at once. Once this
        is exceeded, those experiences that were added first to the
        buffer are removed in the order they were added.

    pretrainLen : int
        The number of initial experiences to fill the buffer with
        before training actually begins. This is used to avoid the
        empty memory problem.

    Methods
    -------
    add(experience)
        Adds the passed experience to the buffer.

    sample(batchSize)
        Extracts a sample of batchSize experiences from the buffer to
        be used in training.

    update()
        Updates the internal state of the memory after a network
        update.
    """

    # -----
    # constructor
    # -----
    def __init__(self, memoryParams):
        """
        Sets up the buffer.

        Parameters
        ----------
        memmoryParams : halsey.utils.folio.Folio
            An object containing the memory-specific data read from the
            parameter file.

        Raises
        ------
        None

        Returns
        -------
        None
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
        Adds the passed experience to the memory's buffer.

        The buffer is a deque, which is a first-in-first-out data
        structure.

        Parameters
        ----------
        experience : halsey.utils.experience.Experience
            Object that holds the Markovian information about an
            agent's interaction with a given state. Contains the
            state encountered, the action chosen, the reward given by
            the game, the next state resulting from the action choice,
            and whether or not the resulting state is a terminal state
            or not.

        Raises
        ------
        None

        Returns
        -------
        None
        """
        self.buffer.append(experience)

    # -----
    # Sample
    # -----
    def sample(self, batchSize):
        """
        Extracts batchSize experiences from the buffer, processes them
        into the form required by the network, and returns the sample.

        Currently, this method selects each experience randomly, with
        each experience having an equal chance to be chosen.
        Prioritized Experience Replay (PER) will be added in the
        future.

        Parameters
        ----------
        batchSize : int
            The number of experiences to extract from the buffer.

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

    # -----
    # update
    # -----
    def update(self):
        """
        Updates the memory's internal state after a network update.

        This mostly applies to updating the importance-sampling weights
        assigned to each experience in the buffer when using PER, which
        has not yet been implemented. So, currently, this method does
        nothing.

        Parameters
        ----------
        None

        Raises
        ------
        None

        Returns
        -------
        None
        """
        pass
