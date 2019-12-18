"""
Title: utils.py
Purpose:
Notes:
    * The brain class contains the neural network
    * The brain class contains the learn method
    * The brain class contains methods for managing and updating the
        neural network
"""
from anna.brains.dql.double import DoubleDqnBrain
from anna.brains.dql.fixed import FixedQBrain
from anna.brains.dql.vanilla import VanillaQBrain


# ============================================
#               get_new_brain
# ============================================
def get_new_brain(brainParams, nActions, inputShape, channelsFirst):
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
    if brainParams.mode == "vanillaQ":
        brain = VanillaQBrain(brainParams, nActions, inputShape, channelsFirst)
    elif brainParams.mode == "fixedQ":
        brain = FixedQBrain(brainParams, nActions, inputShape, channelsFirst)
    elif brainParams.mode == "doubleDqn":
        brain = DoubleDqnBrain(brainParams, nActions, inputShape, channelsFirst)
    return brain
