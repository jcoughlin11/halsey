"""
Title: base.py
Notes:
"""
from abc import ABC, abstractmethod

from halsey.utils.register import register


# ============================================
#               BaseNavigator
# ============================================
class BaseNavigator(ABC):
    """
    The `navigator` object handles choosing actions, taking those
    actions in-game, and then transitioning to the next game state.
    """

    # -----
    # subclass hook
    # -----
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register(cls)

    # -----
    # constructor
    # -----
    def __init__(self, env, explorer, imagePipeline, params):
        self.env = env
        self.explorer = explorer
        self.imagePipeline = imagePipeline
        self.params = params
        self.state = None

    # -----
    # reset
    # -----
    @abstractmethod
    def reset(self):
        """
        Resets the game to its starting state.
        """
        pass

    # -----
    # transition
    # -----
    @abstractmethod
    def transition(self, brain, mode):
        """
        Oversees action selection, taking the action, and moving the
        game to the resulting state.
        """
        pass
