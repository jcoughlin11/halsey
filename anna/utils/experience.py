"""
Title: experience.py
Purpose:
Notes:
"""


# ============================================
#                 Experience
# ============================================
class Experience:
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
    def __init__(self, state, action, reward, nextState, done):
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
        self.state = state
        self.action = action
        self.reward = reward
        self.nextState = nextState
        self.done = done