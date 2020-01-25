"""
Title: basebrain.py
Purpose: Contains the base brain class.
Notes:
"""
import halsey


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
    def __init__(self, brainParams, nActions, inputShape, channelsFirst):
        """
        Initializes the brain and builds the primary network.

        Parameters
        ----------
        brainParams : halsey.utils.folio.Folio
            Contains the brain-related parameters read in from the
            parameter file.

        nActions : int
            The size of the game's action space. Determines the
            network's output shape.

        inputShape : list
            Contains the dimensions of the input to the network.

        channelsFirst : bool
            If True, then the first element of inputShape is the number
            of channels in the input. If False, then the last element of
            inputShape is assumed to be the number of channels.

        Raises
        ------
        None

        Returns
        -------
        None
        """
        # Initialize attributes
        self.arch = brainParams.architecture
        self.inputShape = inputShape
        self.channelsFirst = channelsFirst
        self.nActions = nActions
        self.learningRate = brainParams.learningRate
        self.discountRate = brainParams.discount
        self.optimizerName = brainParams.optimizer
        self.lossName = brainParams.loss
        self.loss = 0.0
        self.qNet = None
        # Build primary network
        self.qNet = halsey.networks.utils.build_network(
            self.arch,
            self.inputShape,
            self.channelsFirst,
            self.nActions,
            self.optimizerName,
            self.lossName,
            self.learningRate,
        )

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
