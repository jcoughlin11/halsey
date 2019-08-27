"""
Title:   utils.py
Author:  Jared Coughlin
Date:    8/27/19
Purpose: Contains helper functions related to setting up the neural
         networks.
"""


# ============================================
#                  set_loss
# ============================================
def set_loss(loss):
    """
    If applicable, assigns the string version of the loss function to
    the actual function version. Some strings, such as 'mse', are ok
    since tf recognizes them. For custom loss functions, such as with
    PER, it must be set explicitly.

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
    # Check for custom loss functions. If we're here, then the loss has
    # already been confirmed to be in the lossRegister and, therefore,
    # valid and listed here
    if loss == "per_mse":
        loss = losses.per_mse
    return loss 


# ============================================
#               set_optimizer
# ============================================
def set_optimizer(optimizer, learningRate):
    """
    Converts the string form of the optimizer to the class form and
    applies the learning rate.

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
    if optimizer == "adam":
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learningRate
        )
    return optimizer 


#============================================
#               init_network
#============================================
def init_network(params):
    """
    Driver function for building the appropriate network for the agent.

    Parameters:
    -----------
        params : dict
            Dictionary of parameters read in from the parameter file.

    Raises:
    -------
        pass

    Returns:
    --------
        pass
    """
    # Set the loss function
    params['loss'] = set_loss(params['loss'])
    # Set the optimizer
    params['optimizer'] = set_optimizer(params['optimizer'], params['learningRate'])
    # Set up the brain object
    if params['agentType'] == 'q':
        brain = anna.brains.qbrain.Brain(params)
    return brain
