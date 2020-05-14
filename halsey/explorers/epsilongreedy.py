"""
Title: epsilongreedy.py
Notes:
"""
from .base import BaseExplorer


# ============================================
#           EpsilonGreedyExplorer
# ============================================
class EpsilonGreedyExplorer(BaseExplorer):
    """
    Doc string.
    """
    # -----
    # constructor
    # -----
    def __init__(self, params):
        """
        Doc string.
        """
        super().__init__(params)
