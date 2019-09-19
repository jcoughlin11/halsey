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
        self.messenger = anna.utils.messenger.Messenger(clArgs, params)
        # Set the relevant io params
        self.ioManager.set_params(self.messenger.ioParams)

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
        # Initialize the game environment
        env = anna.utils.env.init_env(self.messenger.runParams.envName)
        # Set up the brain, memory, and trainer objects
        if self.messenger.runParams.continueTraining:
            brain   = self.ioManager.reader.load_checkpoint_brain()
            memory  = self.ioManager.reader.load_memory()
            trainer = self.ioManager.reader.load_trainer(env)
        else:
            brain   = anna.brains.utils.get_new_brain(self.messenger)
            memory  = anna.memory.utils.get_new_memory(self.messenger)
            trainer = anna.trainers.utils.get_new_trainer(self.messenger, env)
        # Train the network
        while not trainer.done:
            brain, memory = trainer.train(brain, memory)
            if trainer.saveCheckpoint:
                self.ioManager.writer.save_checkpoint(brain, memory, trainer)
        # Save final model if training wasn't ended early
        if not trainer.earlyStop:
            self.ioManager.writer.save_final(brain)
        # Save a copy of the current parameter file that's in use, if
        # it hasn't been saved already (from a previous run)
        self.ioManager.writer.save_param_file()
        # If early stopping, prevent this function from returning to
        # run_agent and therefore continuing the program with potential
        # calls to other methods
        if trainer.earlyStop:
            sys.exit()

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
        # Load the saved final network
        brain = self.ioManager.reader.load_final_brain()
        # Instantiate the tester object
        tester = anna.testers.utils.get_tester()
        # Test the agent
        tester.test(brain)

    #-----
    # plot
    #-----
    def plot(self):
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
        return self.messenger.runParams.train

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
        return self.messenger.runParams.test

    #-----
    # plottingEnabled 
    #-----
    @property
    def plottingEnabled(self):
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
        return self.messenger.runParams.plot
