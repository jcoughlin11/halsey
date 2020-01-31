"""
Title: basechooser.py
Purpose: Provides choosers with access to methods that either randomly
            choose an action or use the network to choose an action.
Notes:
"""
import sys

import numpy as np


# ============================================
#               BaseChooser
# ============================================
class BaseChooser:
    """
    Provides choosers with access to methods that either randomly
    choose an action or use the network to choose an action.

    There are three ways to choose an action: randomly, using the
    network, or using a strategy (e.g., epsilon-greedy). The first two
    options are common to all choosers, so rather than re-implement
    those methods for each chooser, they can inherit from this class.

    Attributes
    ----------
    None

    Methods
    -------
    choose(state, env, brain, mode)
        Abstracts aways specific calls for choosing randomly, with a
        network, or with a strategy.
    """

    # -----
    # choose
    # -----
    def choose(self, state, env, brain, mode):
        """
        Abstracts away specific calls for choosing randomly, with a
        network, or with a strategy.

        Parameters
        ----------
        state : np.ndarray
            The game state being used to inform any decision made with
            the network.

        env : gym.Env
            The interface between the game and the agent.

        brain : halsey.brains.QBrain
            Contains the agent's network(s), learning method, and
            network/learning meta-data.

        mode : str
            Determines how the action is chosen. If `'random'`, then
            the action is chosen randomly. If `'train'`, then the user
            specified action choosing strategy is employed (e.g.,
            epsilon-greedy). If `'test'`, then the brain is used.

        Raises
        ------
        None

        Returns
        -------
        action : int
            The integer value corresponding to the chosen action.
        """
        if mode == "train":
            action = self._train_choose(state, env, brain)
        elif mode == "random":
            action = self._random_choose(env)
        elif mode == "test":
            action = self._test_choose(state, brain)
        else:
            print("Error, unrecognized mode for action selection!")
            sys.exit(1)
        return action

    # -----
    # random_choose
    # -----
    def _random_choose(self, env):
        """
        Chooses a value from [0, nActions) with each value havin an
        equal probability of being selected.

        Parameters
        ----------
        env : gym.Env
            The interface between the game and the agent.

        Raises
        ------
        None

        Returns
        -------
        action : int
            The integer value corresponding to the chosen action.
        """
        action = env.action_space.sample()
        return action

    # -----
    # test_choose
    # -----
    def _test_choose(self, state, brain):
        """
        Uses the brain's current knowledge to select an action.

        Parameters
        ----------
        state : np.ndarray
            The game state being used to inform any decision made with
            the network.

        brain : halsey.brains.QBrain
            Contains the agent's network(s), learning method, and
            network/learning meta-data.

        Raises
        ------
        None

        Returns
        -------
        action : int
            The integer value corresponding to the chosen action.
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
