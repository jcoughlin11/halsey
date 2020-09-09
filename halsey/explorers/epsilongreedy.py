"""
Title: epsilongreedy.py
Notes:
"""
import numpy as np

from .base import BaseExplorer


# ============================================
#           EpsilonGreedyExplorer
# ============================================
class EpsilonGreedyExplorer(BaseExplorer):
    """
    Implements the epsilon-greedy exploration-exploitation strategy.
    """

    # -----
    # constructor
    # -----
    def __init__(self, params):
        super().__init__(params)
        self.params["decayStep"] = 0

    # -----
    # train_choice
    # -----
    def train_choice(self, state, env, brain):
        """
        Uses the chosen exploration-exploitation strategy to choose an
        action.
        """
        exploitProb = np.random.random()
        exploreProb = self.params["epsilonStop"] + (
            self.params["epsilonStart"] - self.params["epsilonStop"]
        ) * np.exp(-self.params["epsDecayRate"] * self.params["decayStep"])
        self.params["decayStep"] += 1
        if exploreProb >= exploitProb:
            action = self.random_choice(env)
        else:
            action = self.test_choice(state, brain)
        return action
