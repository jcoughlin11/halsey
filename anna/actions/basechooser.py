"""
Title: basechooser.py
Purpose:
Notes:
"""
import numpy as np


# ============================================
#               BaseChooser
# ============================================
class BaseChooser:
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
    # choose
    # -----
    def choose(self, state, env, brain, mode):
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
        # Choose an action according to the desired strategy
        if mode == "train":
            action = self.train_choose(state, env, brain)
        elif mode == "random":
            action = self.random_choose(env)
        elif mode == "test":
            action = self.test_choose(state, brain)
        return action

    # -----
    # random_choose
    # -----
    def random_choose(self, env):
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
        action = env.action_space.sample()
        return action

    # -----
    # test_choose
    # -----
    def test_choose(self, state, brain):
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
        # Keras expects a group of samples of the specified shape,
        # even if there's just one sample, so we need to reshape
        state = state.reshape(
            (1, state.shape[0], state.shape[1], state.shape[2])
        )
        # Get the beliefs in each action for the current state
        Q_vals = brain.qNet.model.predict_on_batch(state)
        # Choose the one with the highest Q value
        action = np.argmax(Q_vals)
        return action
