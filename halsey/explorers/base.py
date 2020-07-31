"""
Title: base.py
Notes:
"""
from abc import ABC
from abc import abstractmethod


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
    @abstractmethod
    def choose(self, state, env, brain, mode):
        """
        Driver routine for selecting an action according to `mode`.
        """
        pass

    # -----
    # random_choice
    # -----
    @abstractmethod
    def random_choice(self, env):
        """
        Selects an action randomly.
        """
        pass

    # -----
    # test_choice
    # -----
    @abstractmethod
    def test_choice(self, state, brain):
        """
        Uses the agent's current knowledge to select an action.
        """
        pass

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
