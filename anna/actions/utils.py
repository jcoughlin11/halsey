"""
Title: utils.py
Purpose: Contains functions related to setting up a new action manager.
Notes:
"""
import numpy as np

from anna.actions.epsilongreedy import EpsilonGreedy


# ============================================
#             set_action_manager
# ============================================
def set_action_manager(params, base):
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
    # -----
    # BaseChooser
    # -----
    class BaseChooser(base):
        """
        There are three times when it's necessary to choose an action:
        when pre-populating the memory buffer (actions are randomly
        chosen), when training (actions are chosen according to the
        desired strategy, e.g. epsilon-greedy), and when testing (actions
        are chosen via the best determination of the network). Since
        the procedure for choosing an action in the pre-populating and
        testing cases is the same for every strategy, this class contains
        them.

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
        def __init__(self, params):
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
            super().__init__(params)

        # -----
        # choose
        # -----
        def choose(self, state, env, brain, mode):
            """
            Doc string.

            Parameters:
            -----------
                pass

            Rasies:
            -------
                pass

            Returns:
            --------
                pass
            """
            if mode == "random":
                action = self.random_choose(env)
            elif mode == "train":
                action = self.train_choose(state, env, brain)
            elif mode == "test":
                action = self.test_choose(state, brain)
            else:
                raise ValueError
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

    actionManager = BaseChooser(params)
    return actionManager


# ============================================
#         get_new_action_manager
# ============================================
def get_new_action_manager(exploreParams):
    """
    Doc string. The reason I've done it this way is so that I can have
    dynamic inheritance. I like having one choose function that the
    navigator exposes to the rest of the code. Makes maintaining and
    extending easier. If EpsilonGreedy (or whichever action chooser
    class) inherits from BaseChooser, then it means that every action
    chooser class has to have the same choose method, which maybe isn't
    a big deal, but it's annoying to me. This way, BaseChooser inherits
    from the desired action chooser class, which is determined by the
    user in the parameter file. This means I only need one choose
    method.

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
    if exploreParams.mode == "epsilonGreedy":
        base = EpsilonGreedy
    actionManager = set_action_manager(exploreParams, base)
    return actionManager
