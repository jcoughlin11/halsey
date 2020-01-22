"""
Title: utils.py
Purpose: Handles construction of a new brain object.
Notes:
    * The brain class contains the neural network
    * The brain class contains the learn method
    * The brain class contains methods for managing and updating the
        neural network
"""
from .dql.vanilla import VanillaQBrain


# ============================================
#               get_new_brain
# ============================================
def get_new_brain(brainParams, nActions, inputShape, channelsFirst):
    """
    Handles construction of a new brain object.

    Parameters
    ----------
    brainParams : halsey.utils.folio.Folio
        Contains the brain-related parameters read in from the
        parameter file.

    nActions : int
        The size of the game's action space. Determines the network's
        output shape.

    inputShape : list
        Contains the dimensions of the input to the network.

    channelsFirst : bool
        If True, then the first element of inputShape is the number of
        channels in the input. If False, then the last element of
        inputShape is assumed to be the number of channels.

    Raises
    ------
    None

    Returns
    -------
    brain : halsey.brains.QBrain
    """
    if brainParams.mode == "vanillaQ":
        brain = VanillaQBrain(brainParams, nActions, inputShape, channelsFirst)
    return brain
