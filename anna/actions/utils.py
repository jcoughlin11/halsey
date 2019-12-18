"""
Title: utils.py
Purpose:
Notes:
    * The action manager handles action selection
"""
from .epsilongreedy import EpsilonGreedy


# ============================================
#          get_new_action_manager
# ============================================
def get_new_action_manager(actionParams):
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
    if actionParams.mode == "epsilonGreedy":
        actionManager = EpsilonGreedy(actionParams)
    return actionManager
