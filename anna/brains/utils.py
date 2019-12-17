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
def get_new_brain():
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
    if networkParams.mode == "vanillaQ":
        brain = VanillaQBrain()
    elif networkParams.mode == "fixedQ":
        brain = FixedQBrain()
    elif networkParams.mode == "doubleDqn":
        brain = DoubleDqnBrain()
    return brain
