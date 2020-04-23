"""
Title: base.py
Notes:
"""
import gin

from halsey.utils.setup import get_loss_func
from halsey.utils.setup import get_optimizer


# ============================================
#                 BaseBrain
# ============================================
@gin.configurable(blacklist=["nets"])
class BaseBrain:
    """
    Doc string.

    Attributes
    ----------
    pass

    Methods
    -------
    pass
    """

    # -----
    # constructor
    # -----
    def __init__(self, brainParams, nets=None):
        """
        Doc string.

        NOTE: Should allow for each network in nets to have it's own
        optimizer and loss function (i.e., those should be lists, too)
        When this is done, make sure to update the creation of the
        instructor's checkpoint object
        """
        self.discountRate = brainParams["discountRate"]
        self.learningRate = brainParams["learningRate"]
        self.optimizerName = brainParams["optimizer"]
        self.lossName = brainParams["loss"]
        self.lossFunction = get_loss_func(self.lossName)
        self.optimizer = get_optimizer(self.optimizerName, self.learningRate)
        self.nets = nets
