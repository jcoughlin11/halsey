"""
Title: agent.py
Purpose: Contains the Agent class.
Notes:
    * The Agent class oversees the ioManager, trainer, and tester, and
        the folio.
"""
import halsey


# ============================================
#                  Agent
# ============================================
class Agent:
    """
    Halsey's primary manager class.

    The Agent is responsible for overseeing the setup of each run as
    well as whichever tasks the user wants to run, such as training and
    testing.

    Attributes
    ----------
    folio : halsey.utils.folio.Folio
        A container class for all of the parameters specified in the
        parameter file.

    clArgs : argparse.Namespace
        Container object for all of the command-line arguments.

    ioManager : halsey.io.manager.IoManager
        Object for reading in and saving files.

    Methods
    -------
    setup()
        Oversees the details of initializing a run.

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
        self.ioManager = halsey.io.manager.IoManager()
        self.folio = None
        self.clArgs = None

    # -----
    # setup
    # -----
    def setup(self):
        """
        Handles reading in the command-line arguments, the parameter
        file, parameter validation, creation of the folio object, and
        determines whether or not a GPU is being used.

        The creation of the folio includes the generation of the
        network input shape and the determination of the size of the
        action space so that each section of the folio after this point
        is completely stand-alone.

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
        # Parse the command-line arguments
        self.clArgs = self.ioManager.parse_command_line()
        # Read in the parameter file
        params = self.ioManager.load_parameter_file(self.clArgs.paramFile)
        # Validate the parameters
        halsey.utils.validation.validate_params(params)
        # Create the folio object
        folio = halsey.utils.folio.get_new_folio(params)
        # Set the relevant IO parameters
        self.ioManager.set_io_params(folio.io, self.clArgs.continueTraining)
        # Get the input and output shapes for the network. This is done
        # here because the input shape is also needed by the frame
        # manager and the output shape is also needed by the navigator.
        # This allows for each section of the folio to be stand-alone
        # at the expense of having two copies of each of these
        # variables, but they're small, and the convenience is worth it
        inputShape, nActions = halsey.utils.env.get_shapes(
            folio.brain.architecture, folio.frame, folio.run.envName
        )
        self.folio = halsey.utils.folio.finalize_folio(
            inputShape, nActions, folio
        )

    # -----
    # train
    # -----
    def train(self):
        """
        Manages the primary training loop.

        Training is actually handled by the trainer object. Here we
        instantiate one and then loop over it's train generator, which
        yields when it's time to save a checkpoint file and terminates
        upon training finishing or being ended early by the user.

        Parameters
        ----------
        None

        Raises
        ------
        None

        Returns
        -------
        exitStatus : bool
            Returns True if all training episodes finish, otherwise,
            returns False if training had to end early for any reason.
        """
        # Instantiate objects required for training. Doing it this way
        # is modular, as no code needs to be added here when adding new
        # networks, managers, and the like.
        trainer = halsey.trainers.utils.get_new_trainer(self.folio)
        # Training loop
        for _ in trainer.train():
            self.ioManager.save_checkpoint(trainer)
        # If early stopping, exit
        if trainer.earlyStop:
            print("Training stopped early.")
            exitStatus = False
        else:
            print("Training completed.")
            exitStatus = True
        return exitStatus

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
