"""
Title: epsilongreedy.py
Purpose: Implements the epsilon-greedy exploration-exploitation
            strategy.
Notes:
"""
import logging
import sys

import numpy as np

from halsey.utils.validation import register_option

from .basechooser import BaseChooser


# ============================================
#               EpsilonGreedy
# ============================================
@register_option
class EpsilonGreedy(BaseChooser):
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
    None
    """

    # -----
    # constructor
    # -----
    def __init__(self, actionParams):
        """
        Initializes the epsilon-greedy state.

        Parameters
        ----------
        actionParams : halsey.utils.folio.Folio
            The relevant parameters as read in from the parameter file.

        Raises
        ------
        None

        Returns
        -------
        None
        """
        self.epsDecayRate = actionParams.epsDecayRate
        self.epsilonStart = actionParams.epsilonStart
        self.epsilonStop = actionParams.epsilonStop
        self.decayStep = 0

    # -----
    # train_choose
    # -----
    def _train_choose(self, state, env, brain):
        """
        Implements the epsilon-greedy exploration-exploitation
        strategy.

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

        Raises:
        -------
        None

        Returns:
        --------
        action : int
            The integer value corresponding to the chosen action.
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
        if exploreProb < 0.0:
            infoLogger = logging.getLogger("infoLogger")
            errorLogger = logging.getLogger("errorLogger")
            infoLogger.info("Error: probability of exploring is negative!")
            errorLogger.error("Probability of exploring is negative.")
            sys.exit(1)
        # Explore
        if exploreProb >= exploitProb:
            # Choose randomly
            action = self._random_choose(env)
        # Exploit
        else:
            action = self._test_choose(state, brain)
        return action
