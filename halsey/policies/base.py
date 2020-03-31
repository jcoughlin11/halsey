"""
Title: basechooser.py
Notes:
"""
import numpy as np

from halsey.utils.endrun import endrun


# ============================================
#                 BasePolicy
# ============================================
class BasePolicy:
    """
    Provides access to methods that either randomly choose an action or
    use the network(s) to choose an action.

    Attributes
    ----------
    pass

    Methods
    -------
    pass
    """

    # -----
    # choose
    # -----
    def choose(self, state, env, brain, mode):
        """
        Doc string.
        """
        if mode == "random":
            action = self.random_choose(env)
        elif mode == "train":
            action = self.train_choose(state, env, brain)
        elif mode == "test":
            action = self.test_choose(state, brain)
        else:
            msg = f"Invalid value of mode: `{mode}` in choose()."
            endrun(msg)
        return action

    # -----
    # random_choose
    # -----
    def random_choose(self, env):
        """
        Doc string.
        """
        action = env.action_space.sample()
        return action

    # -----
    # test_choose
    # -----
    def test_choose(self, state, brain):
        """
        Doc string.
        """
        predictions = brain.predict(state)
        action = np.argmax(predictions.numpy())
        return action
