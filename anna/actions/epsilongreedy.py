"""
Title: epsilongreedy.py
Purpose:
Notes:
"""
import numpy as np


# ============================================
#               EpsilonGreedy
# ============================================
class EpsilonGreedy:
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
    def __init__(self, actionParams):
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
        self.epsDecayRate = actionParams.epsDecayRate
        self.epsilonStart = actionParams.epsilonStart
        self.epsilonStop = actionParams.epsilonStop
        self.decayStep = 0

    # -----
    # train_choose
    # -----
    def train_choose(self, state, env, brain):
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
        # Choose a random number from uniform distribution between 0 and
        # 1. This is the probability that we exploit the knowledge we
        # already have
        exploitProb = np.random.random()
        # Get the explore probability. This probability decays over time
        # (but stops at eps_stop so we always have some chance of trying
        # something new) as the agent learns
        exploreProb = self.epsilonStop + (
            self.epsilonStart - self.epsilonStop
        ) * np.exp(-self.epsDecayRate * self.decayStep)
        # Explore
        if exploreProb >= exploitProb:
            # Choose randomly
            action = env.action_space.sample()
        # Exploit
        else:
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
