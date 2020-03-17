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
        """
        self.discountRate = brainParams["discountRate"]
        self.lossFunction = get_loss_func(brainParams["loss"])
        self.optimizer = get_optimizer(
            brainParams["optimizer"], brainParams["learningRate"]
        )
        self.nets = nets
