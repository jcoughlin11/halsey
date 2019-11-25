"""
Title:   baseagent.py
Purpose: Contains the BaseAgent class.
Notes:
"""
import tensorflow as tf

import anna


# ============================================
#                 BaseAgent
# ============================================
class BaseAgent:
    """
    Doc string.

    Attributes:
    -----------
        pass

    Methods:
    --------
        pass
    """

    # -----
    # constructor
    # -----
    def __init__(self):
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
        # Instantiate the ioManager object
        self.ioManager = anna.io.manager.IoManager()
        # Parse the command-line arguments
        clArgs = self.ioManager.reader.parse_cl_args()
        # Read in the parameter file
        params = self.ioManager.reader.read_param_file(
            clArgs.paramFile, clArgs.continueTraining
        )
        # Build messenger object
        self.relay = anna.utils.relay.get_new_relay(clArgs, params)
        # Set the relevant io params
        self.ioManager.set_params(self.relay.io)
        # Set input shape format
        # NOTE: tf does not support the NCHW inputShape format on cpu,
        # but performance is better with channels first on gpu. Also, RNNs
        # require channels first. Cpu needs NHWC
        arch = self.relay.network.architecture
        if not tf.test.is_built_with_gpu_support() and not arch == "rnn1":
            channelsFirst = False
        else:
            channelsFirst = True
        setattr(self.relay.frame, "channelsFirst", channelsFirst)

    # -----
    # trainingEnabled
    # -----
    @property
    def trainingEnabled(self):
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
        return self.relay.run.train

    # -----
    # testingEnabled
    # -----
    @property
    def testingEnabled(self):
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
        return self.relay.run.test
