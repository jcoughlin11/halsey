"""
Title:   brain.py
Purpose: Contains the base Brain class.
Notes:
"""


#============================================
#                  Brain
#============================================
class Brain:
    """
    Container and toolchain for setting up the elements common to all
    artificial brains.

    Attributes:
    -----------
        pass

    Methods:
    --------
        pass
    """
    #-----
    # constructor
    #-----
    def __init__(self, optimizer, loss, learningRate, nActions, inputShape, architecture, backend):
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
        self.optimizer = anna.networks.utils.set_optimizer(optimizer)
        self.loss = anna.networks.utils.set_loss(loss)
        self.learningRate = learningRate
        self.nActions = nActions
        self.inputShape = inputShape
        self.arch = architecture
        self.backend = backend
        # Build the primary network
        self.qnet = anna.networks.utils.build_net(backend, architecture, inputShape, nActions)
        self.qnet.compile(loss=self.loss, optimizer=self.optimizer(lr=self.learningRate))
