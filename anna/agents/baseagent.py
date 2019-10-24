"""
Title:   baseagent.py
Purpose: Contains the BaseAgent class.
Notes:
"""
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
        # Validate command-line args and params
        anna.utils.validation.validate_params(clArgs, params)
        # Build messenger object
        self.relay = anna.utils.relay.get_new_relay(clArgs, params)
        # Set the relevant io params
        self.ioManager.set_params(self.relay.ioParams)

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
        return self.relay.runParams.train

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
        return self.relay.runParams.test
