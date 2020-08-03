"""
Title: base.py
Notes:
"""
from abc import ABC
from abc import abstractmethod

import numpy as np


# ============================================
#               BaseExplorer
# ============================================
class BaseExplorer(ABC):
    """
    The `explorer` object is responsible for action selection.
    """

    # -----
    # constructor
    # -----
    def __init__(self, params):
        self.params = params

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
    @abstractmethod
    def train_choice(self, state, env, brain):
        """
        Uses the chosen exploration-exploitation strategy to choose an
        action.
        """
        pass
