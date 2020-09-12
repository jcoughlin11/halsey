"""
Title: base.py
Notes:
"""
from abc import ABC, abstractmethod

from halsey.utils.register import register


# ============================================
#               BaseInstructor
# ============================================
class BaseInstructor(ABC):
    """
    Template for all instructor classes.

    The `instructor` object contains the training loop.
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
    def __init__(self, brain, navigator, params):
        self.brain = brain
        self.navigator = navigator
        self.params = params

    # -----
    # train
    # -----
    @abstractmethod
    def train(self):
        """
        Contains the main training loop.
        """
        pass
