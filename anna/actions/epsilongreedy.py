"""
Title: epsilongreedy.py
Purpose: Contains the class for using the epsilon-greedy method for
            explore-exploit.
Notes:
"""


#============================================
#               EpsilonGreedy
#============================================
class EpsilonGreedy:
    """
    Doc string.

    Attributes:
    -----------
        pass

    Methods:
    --------
        pass
    """
    #-----
    # constructor
    #-----
    def __init__(self, exploreParams):
        """
        Doc string.

        Parameters:
        -----------
            pass

        Raises:
        -------
            pass

        Returns:
        --------
            pass
        """
        self.epsDecayRate = exploreParams.epsDecayRate
        self.epsilonStart = exploreParams.epsilonStart
        self.epsilonStop = exploreParams.epsilonStop
        self.decayStep = 0

