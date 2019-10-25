"""
Title: utils.py
Purpose: Contains functions related to setting up a new action manager.
Notes:
"""
from anna.actions.epsilongreedy import EpsilonGreedy


# ============================================
#         get_new_action_manager
# ============================================
def get_new_action_manager(exploreParams):
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
    if exploreParams.mode == "epsilonGreedy":
        actionManager = EpsilonGreedy(exploreParams)
    return actionManager
