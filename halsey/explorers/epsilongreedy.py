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
    # choose
    # -----
    def choose(self, state, env, brain, mode):
        """
        Driver routine for selecting an action according to `mode`.
        """
        if mode == "random":
            action = self.random_choice(env)
        elif mode == "train":
            action = self.train_choice(state, env, brain)
        elif mode == "test":
            action = self.test_choice(state, brain)
        return action

    # -----
    # random_choice
    # -----
    def random_choice(self, env):
        """
        Selects an action randomly.
        """
        action = env.action_space.sample()
        return action

    # -----
    # test_choice
    # -----
    def test_choice(self, state, brain):
        """
        Uses the agent's current knowledge to select an action.
        """
        predictions = brain.predict(state)
        action = np.argmax(predictions.numpy())
        return action

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
        if exploreProb >= exploitProb:
            action = self.random_choice(env)
        else:
            action = self.test_choice(state, brain)
        return action
