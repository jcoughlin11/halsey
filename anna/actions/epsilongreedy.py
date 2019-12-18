"""
Title: epsilongreedy.py
Purpose:
Notes:
"""
import numpy as np

from .basechooser import BaseChooser


# ============================================
#               EpsilonGreedy
# ============================================
class EpsilonGreedy(BaseChooser):
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
            action = self.random_choose(env)
        # Exploit
        else:
            action = self.test_choose(state, brain)
        return action
