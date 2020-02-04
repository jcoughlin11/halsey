"""
Title: basebrain.py
Purpose: Contains the base brain class.
Notes:
"""
import logging
import sys

import tensorflow as tf

from halsey.utils.validation import optionRegister


# ============================================
#             set_optimizer
# ============================================
def set_optimizer(optimizerName, learningRate):
    """
    Assigns the actual optimizer function based on the given string
    form.

    This way of doing it allows for not only native keras optimizers,
    but also custom, user-defined optimizers, as well. Both
    user-defined and native keras optimizers are handled in
    the same manner, which makes life simpler, if potentially more
    verbose than necessary in certain cases.

    Parameters
    ----------
    optimizerName : str
        The name of the optimizer to use.

    learningRate : float
        The step size to use during back propagation.

    Raises
    ------
    None

    Returns
    -------
    optimizer : tf.keras.optimizers.Optimizer
        The actual optimizer object to perform minimization of the loss
        function.
    """
    if optimizerName == "adam":
        optimizer = tf.keras.optimizers.Adam(lr=learningRate)
    else:
        infoLogger = logging.getLogger("infoLogger")
        errorLogger = logging.getLogger("errorLogger")
        infoLogger.info("Error: unrecognized optimizer!")
        errorLogger.error("Unrecognized optimizer.")
        sys.exit(1)
    return optimizer


# ============================================
#                set_loss
# ============================================
def set_loss(lossName):
    """
    Sets the actual loss function to be minimized based on the given
    string form.

    This way of doing it allows for not only native keras losses,
    but also custom, user-defined losses, as well. Both
    user-defined and native keras lossesare handled in
    the same manner, which makes life simpler, if potentially more
    verbose than necessary in certain cases.

    Parameters
    ----------
    lossName : str
        The name of the loss function to use.

    Raises
    ------
    None

    Returns
    -------
    loss : tf.keras.losses.Loss
        The actual loss function to be minimized during training.
    """
    if lossName == "mse":
        loss = tf.keras.losses.MeanSquaredError()
    else:
        infoLogger = logging.getLogger("infoLogger")
        errorLogger = logging.getLogger("errorLogger")
        infoLogger.info("Error: unrecognized loss function!")
        errorLogger.error("Unrecognized loss function.")
        sys.exit(1)
    return loss


# ============================================
#                  BaseBrain
# ============================================
class BaseBrain:
    """
    The base brain class.

    This class holds all of the meta-data about the network(s) and
    handles the construction of the primary network.

    Attributes
    ----------
    arch : str
        The name(s) of the architecture(s) used for the network(s).

    channelsFirst : bool
        If True, then the first element of inputShape is the number of
        channels in the input. If False, then the last element of
        inputShape is assumed to be the number of channels.

    discountRate : float
        Determines how much importance the agent gives to future
        rewards.

    inputShape : list
        Contains the dimensions of the input to the network.

    learningRate : float
        Determines the step-size used during backpropogation.

    loss : float
        The value of the loss from the most recent network update.

    lossName : str
        The name of the loss function to use.

    nActions : int
        The size of the game's action space. Determines the network's
        output shape.

    optimizerName : str
        The name of the method used to to optimize the loss
        function.

    qNet : tf.keras.Model
        The primary neural network used by the brain.

    Methods
    -------
    None
    """

    # -----
    # constructor
    # -----
    def __init__(self, brainParams):
        """
        Initializes the brain and builds the primary network.

        Parameters
        ----------
        brainParams : halsey.utils.folio.Folio
            Contains the brain-related parameters read in from the
            parameter file.

        Raises
        ------
        None

        Returns
        -------
        None
        """
        # Initialize attributes
        self.arch = brainParams.architecture
        self.inputShape = brainParams.inputShape
        self.channelsFirst = brainParams.channelsFirst
        self.nActions = brainParams.nActions
        self.learningRate = brainParams.learningRate
        self.discountRate = brainParams.discount
        self.optimizerName = brainParams.optimizer
        self.lossName = brainParams.loss
        self.qNet = None
        # Set the loss function and optimizer
        self.optimizer = set_optimizer(self.optimizerName, self.learningRate)
        self.loss = set_loss(self.lossName)
        # Build primary network
        self.qNet = optionRegister[self.arch](
            self.inputShape, self.channelsFirst, self.nActions,
        )
        self.qNet.compile(optimizer=self.optimizer, loss=self.loss)

    # -----
    # update
    # -----
    def update(self):
        """
        Updates the brain's internal state (counters, non-primary
        networks, etc.).

        For vanilla Q-learning, there's nothing to update. This method
        is called by the trainer, which is agnostic to the learning
        method being used, which is why the method is needed here.

        Parameters
        ----------
        None

        Raises
        ------
        None

        Returns
        -------
        None
        """
        pass
