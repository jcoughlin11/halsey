"""
Title:   utils.py
Purpose: Contains functions related to assissting in setting up a new
            brain object.
Notes:
"""
from . import double_dqn_brain
from . import fixed_q_brain
from . import vanilla_q_brain 


#============================================
#              get_new_brain
#============================================
def get_new_brain(networkParams):
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
    if networkParams.mode == 'vanillaQ':
        brain = vanilla_q_brain.Brain(networkParams)
    elif networkParams.mode == 'fixedQ':
        brain = fixed_q_brain.Brain(networkParams)
    elif networkParams.mode == 'doubleDqn':
        brain = double_dqn_brain.Brain(networkParams)
    else:
        raise ValueError("Unrecognized brain type.")
    return brain
