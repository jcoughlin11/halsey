"""
Title: agent.py
Purpose:
Notes:
"""
import anna


#============================================
#                  Agent
#============================================
class Agent:
    """
    Doc string.

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
        # Instantiate the io manager
        self.ioManager = anna.io.manager.IoManager()
        # Read the parameter file and command-line options
        self.folio = self.ioManager.load_params() 
        # Save a copy of the parameters for the run
        self.ioManager.save_params(self.folio)

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
        # If continuing training, load the checkpoint files
        if self.folio.clArgs.continueTraining:
            pass
        # Otherwise, instantiate a new trainer
        else:
            trainer = anna.trainers.utils.get_new_trainer()
        # Training loop
        while not trainer.doneTraining:
            trainer.train()
            self.ioManager.save_checkpoint()
        # Clean up 
        trainer.cleanup()
        self.cleanup()
        # If early stopping, exit
        if trainer.earlyStop:
            return False
        else:
            return True

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
    # cleanup
    #-----
    def cleanup(self):
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
        pass

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
        pass