"""
Title: basememory.py
Purpose: Contains the BaseMemory class.
Notes:
"""


# ============================================
#                BaseMemory
# ============================================
class BaseMemory:
    """
    A convenience object for holding all attributes and methods common
    to every memory type.

    At its heart, every memory object is just a way of storing and
    extracting experiences in some way.

    Attributes
    ----------
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

    update()
        Updates the internal state of the memory after a network
        update.
    """

    # -----
    # constructor
    # -----
    def __init__(self, memoryParams):
        """
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
