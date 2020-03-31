"""
Title: epsilongreedy.py
Notes:
"""
import gin
import numpy as np

from halsey.utils.endrun import endrun

from .base import BasePolicy


# ============================================
#               EpsilonGreedy
# ============================================
@gin.configurable
class EpsilonGreedy(BasePolicy):
    """
    Implements the epsilon-greedy exploration-exploitation strategy.

    In this strategy we use a randomly chosen action with probability
    :math:`\epsilon` and exploit the network's knowledge with
    probability :math:`1-\epsilon`.

    As the network gets better, we want it to utilize its own knowlege
    more frequently, so :math:`\lim_{t\rightarrow \infty} \epsilon = 0`.

    However, we don't want there to ever be a zero probability of trying
    something new and therefore potentially missing out on a better
    choice of action for a given state, so we truncate the exploration
    probability at some value :math:`\epsilon_f`. This means our
    probability of exploring is:

    .. math::

        \epsilon = \epsilon_f + (\epsilon_0 - \epsilon_f)\exp{-\lambda i}

    Where :math:`\epsilon_0` is the starting value of :math:`\epsilon`,
    :math:`\lambda` is the decay rate, and :math:`i` is the number of
    steps the agent has taken in the environment so far (i.e., time
    spent playing).

    Attributes
    ----------
    decayStep : int
        The number of steps the agent has taken in the game.

    epsDecayRay : float
        How quickly :math:`\epsilon` changes from
        :math:`\epsilon_0 \rightarrow \epsilon_f`.

    epsilonStart : float
        The initial value of :math:`\epsilon`.

    epsilonStop : float
        The asymptotic value of :math:`\epsilon`.

    Methods
    -------
    pass
    """

    # -----
    # constructor
    # -----
    def __init__(self, actionParams):
        """
        Initializes the epsilon-greedy state.

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
        self.epsDecayRate = actionParams["epsDecayRate"]
        self.epsilonStart = actionParams["epsilonStart"]
        self.epsilonStop = actionParams["epsilonStop"]
        self.decayStep = 0

    # -----
    # train_choose
    # -----
    def train_choose(self, state, env, brain):
        """
        Implements the epsilon-greedy exploration-exploitation
        strategy.

        Parameters
        ----------
        pass

        Raises:
        -------
         pass

        Returns:
        --------
        pass
        """
        exploitProb = np.random.random()
        exploreProb = self.epsilonStop + (
            self.epsilonStart - self.epsilonStop
        ) * np.exp(-self.epsDecayRate * self.decayStep)
        if exploreProb < 0.0:
            msg = f"Probability of exploring is negative: `{exploreProb}`"
            endrun(msg)
        if exploreProb >= exploitProb:
            action = self.random_choose(env)
        else:
            action = self.test_choose(state, brain)
        return action
