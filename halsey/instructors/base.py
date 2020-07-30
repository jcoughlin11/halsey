"""
Title: base.py
Notes:
"""
from abc import ABC
from abc import abstractmethod


# ============================================
#               BaseInstructor
# ============================================
class BaseInstructor(ABC):
    """
    Template for all instructor classes.

    The `instructor` object contains the training loop.
    """

    # -----
    # constructor
    # -----
    def __init__(self, brain, navigator, params):
        self.brain = brain
        self.navigator = navigator
        self.params = params
        # The neural networks themselves cannot be built until the
        # input and output shapes are known. These come from the
        # ImagePipeline and environment, respectively. Rather than
        # pass these arguments through multiple setup functions in
        # setup_instructor, it's done here
        for i in range(len(self.brain.nets)):
            self.brain.nets[i].build_arch(
                self.navigator.imagePipeline.inputShape,
                self.navigator.env.action_space.n
            )

    # -----
    # train
    # -----
    @abstractmethod
    def train(self):
        """
        Contains the main training loop.
        """
        pass
