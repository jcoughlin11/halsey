"""
Title:   losses.py
Author:  Jared Coughlin
Date:    8/5/19
Purpose: Contains custom loss functions
Notes:
    1.) Keras losses docs: https://keras.io/losses/
    2.) All custom loss functions must take two args: y_true and y_pred
    3.) Custom loss functions must return a scalar for each data point
    4.) For losses requiring extra args (in addition to y_true and
        y_pred), use function closure (a wrapper):
        https://tinyurl.com/y4tfkoo5
    5.) Both y_true and y_pred must be tensors
"""
from tensorflow.keras import backend as K


#============================================
#                  per_mse
#============================================
def per_mse(isWeights):
    """
    The Mean-Squared-Error loss function (MSE) as modified by
    Prioritized-Experience-Replay (PER). See Schaul et al. 2016.

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
    def loss_func(y_true, y_pred):
        # The axis = -1 gives us one value per data point (batch sample)
        # instead of just one value for the whole operation, which is
        # what axis = None does
        loss = K.mean(isWeights * K.square(y_true - y_pred), axis=-1)
        return loss
    return loss_func
