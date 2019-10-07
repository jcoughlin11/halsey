"""
Title:   utils.py
Purpose: Contains functions related to creating a new brain object.
Notes:
"""


# ============================================
#               get_new_brain
# ============================================
def get_new_brain(networkParams, nActions, inputShape):
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
        brain = VanillaQBrain(networkParams, nActions, inputShape)
    elif networkParams.mode == "fixedQ":
        brain = FixedQBrain(networkParams, nActions, inputShape)
    elif networkParams.mode == "doubleDqn":
        brain = DoubleDqnBrain(networkParams, nActions, inputShape)
    return brain
