"""
Title:   experience_memory.py
Purpose: Contains the object that holds and manages the vanilla
            Q-learning experience buffer.
Notes:
"""


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
            experience = navigator.transition(mode='random')
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
            batch_size : int
                The size of the sample to be returned.
        Raises:
        -------
            pass
        Returns:
        --------
            sample : list
                A list of randomly chosen experience tuples from the
                buffer. Chosen without replacement. The length of the
                list is batch_size.
        """
        # Choose random indices from the buffer. Make sure the
        # batch_size isn't larger than the current buffer size or np
        # will complain
        try:
            indices = np.random.choice(
                np.arange(len(self.buffer)), size=batch_size, replace=False
            )
        except ValueError:
            raise (
                "Error, need batch_size < buf_size when sampling from memory!"
            )
        return [self.buffer[i] for i in indices]
