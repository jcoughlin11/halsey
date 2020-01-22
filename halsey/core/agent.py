"""
Title: agent.py
Purpose: Contains the Agent class.
Notes:
    * The Agent class oversees the ioManager, trainer, and tester
"""
import halsey


# ============================================
#                  Agent
# ============================================
class Agent:
    """
    Halsey's primary manager class.

    The Agent class is Halsey's user-facing class and exposes the train
    and test methods.

    Attributes
    ----------
    folio : halsey.utils.folio.Folio
        A container class for all of the parameters specified in the
        parameter file.

    ioManager : halsey.io.manager.IoManager
        Object for reading in and saving files.

    Methods
    -------
    train()
        Contains the primary training loop for the agent.

    trainingEnabled()
        A class property that returns True if the training flag is set
        in the parameter file and False otherwise.
    """

    # -----
    # constructor
    # -----
    def __init__(self):
        """
        Reads in the parameter file, creates the folio, and determines
        which device we're training on (CPU or GPU).

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
        # Instantiate the io manager
        self.ioManager = halsey.io.manager.IoManager()
        # Read the parameter file and command-line options
        self.folio, params = self.ioManager.load_params()
        # Save a copy of the run's parameters
        self.ioManager.save_params(params)
        # Set up the input shape
        channelsFirst = halsey.utils.gpu.set_channels(
            self.folio.brain.architecture
        )
        setattr(self.folio.frame, "channelsFirst", channelsFirst)

    # -----
    # train
    # -----
    def train(self):
        """
        Manages the primary training loop.

        Training is actually handled by the trainer object. Here we
        instantiate one and then loop over it's train generator, which
        yields when it's time to save a checkpoint file and terminates
        upon training finishing.

        Parameters
        ----------
        None

        Raises
        ------
        None

        Returns
        -------
        bool
            Returns True if all training episodes finish, otherwise,
            returns False if training had to end early for any reason.
        """
        # Instantiate a new trainer
        trainer = halsey.trainers.utils.get_new_trainer(self.folio)
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
        Exposes the folio.run.train attribute to the user.

        Parameters
        ----------
        None

        Raises
        ------
        None

        Returns
        -------
        bool
            Returns True if the train flag is set in the parameter file
            and False otherwise.
        """
        return self.folio.run.train
