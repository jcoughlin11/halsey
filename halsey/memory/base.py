"""
Title:      base.py
Purpose:    Contains the BaseMemory class.
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
    maxSize : int
        The largest number of experiences to store at once. Once this
        is exceeded, experiences are removed from the buffer in the
        order they were added.

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
        update (e.g., update the imporance sampling weights).
    """

    # -----
    # constructor
    # -----
    def __init__(self, memoryParams):
        """
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
        self.batchSize = memoryParams["batchSize"]
        self.maxSize = memoryParams["maxSize"]
        self.pretrainLen = memoryParams["pretrainLen"]

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
        experience : tuple
            Tuple that holds the Markovian information about an
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
        Updates the memory's internal state after a network update
        (e.g., the importance sampling weights).

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
