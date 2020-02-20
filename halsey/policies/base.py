"""
Title:      base.py
Purpose:    Contains the BasePolicy class.
Notes:
"""
import numpy as np

from halsey.utils.endrun import endrun


# ============================================
#                 BasePolicy
# ============================================
class BasePolicy:
    """
    Provides policy objects with access to methods that either randomly
    choose an action or use the network to choose an action.

    There are three ways to choose an action: randomly, using the
    network, or using a strategy (e.g., epsilon-greedy). The first two
    options are common to all policies, so rather than re-implement
    those methods for each chooser, they can inherit from this class.

    Attributes
    ----------
    None

    Methods
    -------
    choose(state, env, model, mode)
        Abstracts aways specific calls for choosing randomly, with a
        network, or with a strategy.
    """

    # -----
    # choose
    # -----
    def choose(self, state, env, model, mode):
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

        model : halsey.models.BaseModel
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
            action = self._train_choose(state, env, model)
        elif mode == "random":
            action = self._random_choose(env)
        elif mode == "test":
            action = self._test_choose(state, model)
        else:
            msg = f"Unrecognized mode `{mode}` for action selection."
            endrun(ValueError, msg)
        return action

    # -----
    # random_choose
    # -----
    def _random_choose(self, env):
        """
        Chooses a value from [0, nActions) with each value having an
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
    def _test_choose(self, state, model):
        """
        Uses the model's current knowledge to select an action.

        Parameters
        ----------
        state : np.ndarray
            The game state being used to inform any decision made with
            the network.

        model : halsey.models.BaseModel
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
        # The batch size must always be the first element of the input
        # shape, even when there's only one sample
        state = state.reshape([1] + list(state.shape))
        # Get the beliefs in each action for the current state
        predictions = model.primaryNet(state)
        # Choose the one with the highest value
        action = np.argmax(predictions)
        return action
