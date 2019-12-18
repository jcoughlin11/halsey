"""
Title: qbrain.py
Purpose:
Notes:
"""
import anna


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
    def __init__(self, brainParams, nActions, inputShape, channelsFirst):
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
        self.arch = brainParams.architecture
        self.inputShape = inputShape
        self.channelsFirst = channelsFirst
        self.nActions = nActions
        self.learningRate = brainParams.learningRate
        self.discountRate = brainParams.discount
        self.optimizerName = brainParams.optimizer
        self.lossName = brainParams.loss
        self.loss = 0.0
        self.qNet = None
        # Build primary network
        self.qNet = anna.networks.utils.build_network(
            self.arch,
            self.inputShape,
            self.channelsFirst,
            self.nActions,
            self.optimizerName,
            self.lossName,
            self.learningRate,
        )
