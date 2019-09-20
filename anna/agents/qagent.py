"""
Title:   qagent.py
Purpose: Contains the Agent class for using Q-learning techniques
Notes:
"""
import anna


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
        self.ioManager = anna.nnio.manager.IoManager()
        # Parse the command-line arguments
        clArgs = self.ioManager.reader.parse_cl_args()
        # Read in the parameter file
        params = self.ioManager.reader.read_param_file(clArgs.paramFile, clArgs.continueTraining)
        # Validate command-line args and params
        anna.utils.validation.validate_params(clArgs, params)
        # Build messenger object
        self.relay = anna.utils.relay.Relay(clArgs, params)
        # Set the relevant io params
        self.ioManager.set_params(self.relay.ioParams)

    #-----
    # train
    #-----
    def train(self):
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
        # Initialize game environment
        env = anna.navigation.utils.init_env(self.relay.runParams.envName)
        # If continuing training, load the checkpoint files
        if self.relay.runParams.continueTraining:
            pass
        # Otherwise, instantiate new objects
        else:
            brain = anna.brains.utils.get_new_brain(self.relay.networkParams, env.action_space.n, self.relay.frameParams)
            memory = anna.memory.utils.get_new_memory(self.relay.memoryParams, self.relay.trainingParams.batchSize)

    #-----
    # test
    #-----
    def test(self):
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
        pass

    #-----
    # trainingEnabled 
    #-----
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

    #-----
    # testingEnabled 
    #-----
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
