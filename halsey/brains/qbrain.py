"""
Title: qbrain.py
Notes:
"""
from .base import BaseBrain


# ============================================
#                   QBrain
# ============================================
class QBrain(BaseBrain):
    """
    Contains the learning method presented in Mnih et al. 2013.
    """

    # -----
    # learn
    # -----
    def learn(self):
        """
        The learning method by which the neural network weights are
        updated.
        """
        pass

    # -----
    # predict
    # -----
    def predict(self, state):
        """
        Uses the current knowledge of the neural network(s) to select
        what it thinks is the best action for the current situation.
        """
        pass
