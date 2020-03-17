"""
Title: framenavigator.py
Notes:
"""
import gin

from .base import BaseNavigator


# ============================================
#              FrameNavigator
# ============================================
@gin.configurable
class FrameNavigator(BaseNavigator):
    """
    Represents a fully Markovian process whereby the agent takes in and
    processes each and every frame of the game.

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
        Sets up the navigator object.

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
        super().__init__(env, policy, pipeline)

    # -----
    # transition
    # -----
    def transition(self, brain, mode):
        """
        Doc string.
        """
        action = self.policy.choose(self.state, self.env, brain, mode)
        nextFrame, reward, done, _ = self.env.step(action)
        nextState = self.pipeline.process(nextFrame, False)
        experience = (self.state, action, reward, nextState, done)
        self.state = nextState
        return experience
