"""
Title: framenav.py
Notes:
"""
from .base import BaseNavigator


# ============================================
#               BaseNavigator
# ============================================
class FrameNavigator(BaseNavigator):
    """
    Moves through the game one frame at a time.
    """

    # -----
    # reset
    # -----
    def reset(self):
        """
        Resets the game to its starting state.
        """
        startFrame = self.env.reset()
        self.state = self.imagePipeline.process(startFrame, True)

    # -----
    # transition
    # -----
    def transition(self, brain, mode):
        """
        Oversees action selection, taking the action, and moving the
        game to the resulting state.
        """
        action = self.explorer.choose(self.state, self.env, brain, mode)
        nextFrame, reward, done, _ = self.env.step(action)
        nextState = self.imagePipeline.process(nextFrame, False)
        experience = (self.state, action, reward, nextState, done)
        self.state = nextState
        return experience
