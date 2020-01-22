"""
Title: utils.py
Purpose: Handles construction of a new action manager object.
Notes:
    * The action manager handles action selection
"""
from .epsilongreedy import EpsilonGreedy


# ============================================
#          get_new_action_manager
# ============================================
def get_new_action_manager(actionParams):
    """
    Handles construction of a new action manager object.

    Parameters:
    -----------
    actionParams : halsey.utils.folio.Folio
        The relevant parameters as read in from the parameter file.

    Raises:
    -------
    None

    Returns:
    --------
    actionManager : halsey.actions.BaseChooser
        The object used for choosing actions while interacting with
        the game.
    """
    if actionParams.mode == "epsilonGreedy":
        actionManager = EpsilonGreedy(actionParams)
    return actionManager
