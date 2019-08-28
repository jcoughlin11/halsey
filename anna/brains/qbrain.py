"""
Title: qbrain.py
Author: Jared Coughlin
Date: 8/27/19
Purpose: Contains the Brain class for use with a qAgent
Notes:
"""


#============================================
#                   Brain
#============================================
class Brain:
    """
    Holds the agent's network(s) and handles learning.

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
    def __init__(self, params):
        """
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
        self.qNet = None
        self.tNet = None
        # Set the loss function
        params['loss'] = anna.networks.utils.set_loss(params['loss'])
        # Set the optimizer
        params['optimizer'] = anna.networks.utils.set_optimizer(params['optimizer'], params['learningRate'])
        # Build the primary network
        if params['architecture'] == 'conv1':
            self.qNet = anna.networks.conv1.build_conv1_net(params)
        elif params['architecture'] == 'dueling1':
            self.qNet = anna.networks.dueling1.build_dueling1_net(params)
        elif params['architecture'] == 'rnn1':
            self.qNet = anna.networks.rnn1.build_rnn1_net(params)
        else:
            raise ValueError("Unrecognized network architecture!")
        # Compile the model
        self.qNet.compile(optimizer=params['optimizer'], loss=params['loss'])
        # Build target network, if applicable
        if params['enableFixedQ']:
            # Copy layer structure
            self.tNet = tf.keras.models.clone_model(self.qNet)
            # Copy weights
            self.tNet.set_weights(self.qNet.get_weights())
            # Compile the model
            self.tNet.compile(optimizer=params['optimizer'], loss=params['loss'])
