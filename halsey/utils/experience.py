"""
Title: experience.py
Purpose: Contains the Experience object.
Notes:
"""


# ============================================
#                 Experience
# ============================================
class Experience:
    """
    This object stores and manages the information related to an
    agent's interaction with the game.

    For a traditional Markovian interaction, this class simply
    provides an object interface to the (state, action, reward,
    next state, done) tuple.

    Attributes
    ----------
    action : int
        The action chosen for the interaction.

    done : bool
        Whether or not the chosen action led to a terminal state or
        not.

    nextState : np.ndarray
        The state the chosen action results in.

    reward : int
        The feedback provided by the game for having chosen the
        selected action.

    state : np.ndarray
        The state the agent interacted with.

    Methods
    -------
    None
    """

    # -----
    # constructor
    # -----
    def __init__(self, state, action, reward, nextState, done):
        """
        Parameters
        ----------
        state : np.ndarray
            The state the agent interacted with.

        action : int
            The action chosen for the interaction.

        reward : int
            The feedback provided by the game for having chosen the
            selected action.

        nextState : np.ndarray
            The state the chosen action results in.

        done : bool
            Whether or not the chosen action led to a terminal state or
            not.

        Raises
        ------
        None

        Returns
        -------
        None
        """
        self.state = state
        self.action = action
        self.reward = reward
        self.nextState = nextState
        self.done = done
