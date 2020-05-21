"""
Title: epsilongreedy.py
Notes:
"""


# ============================================
#           EpsilonGreedyExplorer
# ============================================
class EpsilonGreedyExplorer:
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
        self.epsDecayRate = params["epsDecayRate"]
        self.epsilonStart = params["epsilonStart"]
        self.epsilonStop = params["epsilonStop"]
