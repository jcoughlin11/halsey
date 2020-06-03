"""
Title: base.py
Notes:
"""
import numpy as np

from halsey.utils.endrun import endrun


# ============================================
#                  BaseGame
# ============================================
class BaseGame:
    """
    Doc string.
    """

    # -----
    # constructor
    # -----
    def __init__(self, env, explorer, pipeline, params):
        """
        Doc string.
        """
        self.env = env
        self.explorer = explorer
        self.pipeline = pipeline
        self.params = params
        self.frameStack = None

    # -----
    # reset
    # -----
    def reset(self):
        """
        Doc string.
        """
        frame = self.env.reset()
        self.frameStack = self.pipeline.process(frame, True)

    # -----
    # transition
    # -----
    def transition(self, brain=None, mode="train"):
        """
        Doc string.
        """
        action = self.choose_action(brain, mode)
        nextFrame, reward, done, _ = self.env.step(action)
        nextFrameStack = self.pipeline.process(nextFrame, False)
        experience = (self.frameStack, action, reward, nextFrameStack, done)
        self.frameStack = nextFrameStack
        return experience

    # -----
    # choose_action
    # -----
    def choose_action(self, brain=None, mode="train"):
        """
        Doc string.
        """
        if mode == "train":
            action = self.explorer.choose(brain, self.env, self.frameStack)
        elif mode == "test":
            action = np.argmax(brain.predict(self.frameStack).numpy())
        elif mode == "random":
            action = self.env.action_space.sample()
        else:
            msg = f"Unrecognized mode: `{mode}` for choosing action."
            endrun(msg)
        return action
