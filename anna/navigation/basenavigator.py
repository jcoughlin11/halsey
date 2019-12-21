"""
Title: basenavigator.py
Purpose: Contains the BaseNavigator class
Notes:
"""


# ============================================
#               BaseNavigator
# ============================================
class BaseNavigator:
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
    def __init__(self, navParams, env, frameManager, actionManager):
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
        self.env = env
        self.frameManager = frameManager
        self.actionManager = actionManager
        self.state = None

    # -----
    # reset
    # -----
    def reset(self):
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
        # Reset the game environment
        state = self.env.reset()
        # Process the initial frame
        self.state = self.frameManager.process_frame(state, newEpisode=True)
