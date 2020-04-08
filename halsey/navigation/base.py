"""
Title: basenavigator.py
Notes:
"""


# ============================================
#               BaseNavigator
# ============================================
class BaseNavigator:
    """
    The navigator represents how the agent interacts with the game
    environment.

    Attributes
    ----------
    pass

    Methods
    -------
    pass
    """

    # -----
    # constructor
    # -----
    def __init__(self, env, policy, pipeline):
        """
        Initializes the object.

        Parameters
        ----------
        pass

        Raises
        ------
        pass

        Returns
        -------
        pass
        """
        self.env = env
        self.policy = policy
        self.pipeline = pipeline
        self.state = None
        self.navParams = None

    # -----
    # reset
    # -----
    def reset(self):
        """
        Doc string.
        """
        startFrame = self.env.reset()
        self.state = self.pipeline.process(startFrame, True)
