"""
Title: agent.py
Purpose:
Notes:
    * The Agent class oversees the ioManager, trainer, and tester
"""
import anna


# ============================================
#                  Agent
# ============================================
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
        # Instantiate the io manager
        self.ioManager = anna.io.manager.IoManager()
        # Read the parameter file and command-line options
        self.folio, params = self.ioManager.load_params()
        # Save a copy of the run's parameters
        self.ioManager.save_params(params)
        # Set up the input shape
        channelsFirst = anna.utils.gpu.set_channels(
            self.folio.brain.architecture
        )
        setattr(self.folio.frame, "channelsFirst", channelsFirst)

    # -----
    # train
    # -----
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
            trainer = self.ioManager.load_checkpoint()
        # Otherwise, instantiate a new trainer
        else:
            trainer = anna.trainers.utils.get_new_trainer(self.folio)
        # Training loop
        for _ in trainer.train():
            self.ioManager.save_checkpoint(trainer)
        # If early stopping, exit
        if trainer.earlyStop:
            print("Training stopped early.")
            return False
        else:
            print("Training completed.")
            return True

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
        return self.folio.run.train
