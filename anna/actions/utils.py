"""
Title: utils.py
Purpose:
Notes:
    * The action manager handles action selection
"""
import anna


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
        actionManager = anna.actions.epsilongreedy.EpsilonGreedy(actionParams)
    return actionManager
