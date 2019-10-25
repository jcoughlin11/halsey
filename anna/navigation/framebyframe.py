"""
Title: framebyframe.py
Purpose: Class for stepping through the game world one frame at a time.
Notes:
"""
from anna.navigation.basenavigator import BaseNavigator
from anna.utils.experience import Experience


# ============================================
#          FrameByFrameNavigator
# ============================================
class FrameByFrameNavigator(BaseNavigator):
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
    def __init__(self, env, navParams, frameManager, actionManager):
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
        super().__init__(env, navParams, frameManager, actionManager)

    # -----
    # transition
    # -----
    def transition(self, brain=None, mode="train"):
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
        # Choose an action
        action = self.actionManager.choose(self.state, brain, mode)
        # Take the action
        nextState, reward, done, _ = self.env.step(action)
        # Process the next state
        nextState = self.frameManager.process_frame(nextState)
        # Package up the experience
        experience = Experience(self.state, action, reward, nextState, done)
        return experience
