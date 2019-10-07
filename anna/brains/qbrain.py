"""
Title: qbrain.py
Purpose: Contains the base Brain class for Q-learning techniques.
Notes:
"""


# ============================================
#                   QBrain
# ============================================
class QBrain:
    """
    Doc string.

    Attributes:
    -----------
        pass

    Methods:
    --------
        pass
    """

    # -----
    # constructor
    # -----
    def __init__(self, networkParams, nActions, inputShape):
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
        # Initialize attributes
        self.arch = networkParams.architecture
        self.inputShape = inputShape
        self.nActions = nActions
        self.learningRate = networkParams.learningRate
        self.discountRate = retworkParams.discountRate
        self.optimizerName = networkParams.optimizer
        self.lossName = networkParams.loss
        self.qNet = None
        # Build primary network
        self.qNet = anna.networks.utils.build_network(
            self.arch,
            self.inputShape,
            self.nActions,
            self.optimizerName,
            self.lossName,
            self.learningRate,
        )
