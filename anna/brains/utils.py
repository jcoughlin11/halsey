"""
Title:   utils.py
Purpose: Contains functions related to creating a new brain object.
Notes:
"""


#============================================
#               get_new_brain
#============================================
def get_new_brain(networkParams, nActions, frameParams):
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
    # Get the input shape. When passing data to the network, this is
    # batch size x trace length x n rows x n cols. For building the
    # nets, we omit batch size, since that's variable. Having the
    # trace length first makes it inherently compatible with RNNs
    inputShape = (frameParams.traceLen, frameParams.shrinkRows, frameParams.shrinkCols)
    if networkParams.mode == 'vanillaQ':
        brain = VanillaQBrain(networkParams, nActions, inputShape)
    elif networkParams.mode == 'fixedQ':
        brain = FixedQBrain(networkParams, nActions, inputShape)
    elif networkParams.mode == 'doubleDqn':
        brain = DoubleDqnBrain(networkParams, nActions, inputShape)
    return brain
