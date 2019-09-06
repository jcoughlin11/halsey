"""
Title:   qagent.py
Purpose: Contains the Agent class for using Q-learning techniques
Notes:
"""
from anna.nnio.manager import IoManager


#============================================
#                   Agent
#============================================
class Agent:
    """
    The primary object manager for using Q-learning techniques.

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
        self.ioManager = IoManager()
        # Parse the command-line arguments
        clArgs = self.ioManager.parse_cl_args()
        # Read in the parameter file
        params = self.ioManager.reader.read_param_file(clArgs.paramFile)
        # Validate the params and command-line args
        utils.validation.validate_params(params, clArgs)
