"""
Title:   manager.py
Purpose: Contains the IoManager class for abstracting away reading and
            writing files.
Notes:
    * The IoManager class oversees the reader and writer objects
"""
import os

import halsey

from .reader import Reader
from .writer import Writer


# ============================================
#                 IoManager
# ============================================
class IoManager:
    """
    Convenience object for managing the reading and writing of files.

    Attributes
    ----------
    brainDir : str
        The full path to where the brain object is saved.

    fileBase : str
        The prefix used for the output file names.

    outputDir : str
        The name of the parent output directory.

    reader : halsey.io.Reader
        The object responsible for reading in all files and parsing
        command-line arguments.

    writer : halsey.io.Writer
        The object responsible for saving all output files.

    Methods
    -------
    parse_command_line()
        Abstracts the reading of the passed command-line options to the
        reader object.

    load_parameter_file()
        Abstracts the reading of the parameter file to the reader
        object.

    save_checkpoint(trainer)
        Saves the current state of the trainer.

    save_params(params)
        Saves a copy of the parameters used during the run.
    """

    # -----
    # constructor
    # -----
    def __init__(self):
        """
        Instantiates paths and creates the reader and writer objects.

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
        self.reader = Reader()
        self.writer = Writer()
        self.fileBase = None
        self.outputDir = None
        self.brainDir = None

    # -----
    # parse_command_line
    # -----
    def parse_command_line(self):
        """
        Parses any optional command-line arguments.

        This is a convenience method that abstracts all of the reading
        away to the reader object.

        Parameters
        ----------
        None

        Raises
        ------
        None

        Returns
        -------
        clArgs : argparse.Namespace
            An object containing all of the available command-line
            arguments and their values.
        """
        clArgs = self.reader.parse_args()
        return clArgs

    # -----
    # load_parameter_file
    # -----
    def load_parameter_file(self, paramFile):
        """
        Abstracts away the reading of the parameter file to the reader
        object.

        Parameters
        ----------
        paramFile : str
            The name of the parameter file to read. Includes the path
            to the file. Must be in yaml format.


        Raises
        ------
        None

        Returns
        -------
        params : dict
            The data read in from the parameter file.
        """
        params = self.reader.read_param_file(paramFile)
        return params

    # -----
    # save_params
    # -----
    def save_params(self, params):
        """
        Saves a copy of the parameters used in the run.

        The purpose of this is so that the user has an easy reference
        to which parameters were used. It also makes continuing the
        training process later on much easier. The file is saved as a
        read-only lock file.

        Parameters
        ----------
        params : dict
            The parameters as read in from the given parameter file.

        Raises
        ------
        None

        Returns
        -------
        None
        """
        self.writer.save_params(params, self.outputDir)

    # -----
    # save_checkpoint
    # -----
    def save_checkpoint(self, trainer):
        """
        Saves the current state of the trainer object.

        Parameters
        ----------
        trainer : halsey.trainers.Trainer
            The object that oversees the training process. It contains
            the brain, memory, and navigator.

        Raises
        ------
        None

        Returns
        -------
        None
        """
        # Save the models
        self.writer.save_models(trainer.brain, self.brainDir)

    # -----
    # set_io_params
    # -----
    def set_io_params(self, ioParams, continueTraining):
        """
        Sets up the file paths and creates the output file directory
        tree.

        Parameters
        ----------
        ioParams : halsey.utils.folio.Folio
            An object containing the relevant IO parameters from the
            parameter file.

        continueTraining : bool
            If True, we are using an existing checkpoint file as a
            starting point and resuming the training process from
            where it left off. If False, we're starting training from
            the beginning.

        Raises
        ------
        None

        Returns
        -------
        None
        """
        # Set the names of the various output directories
        self.fileBase = ioParams.fileBase
        self.outputDir = os.path.abspath(os.path.expanduser(ioParams.outputDir))
        self.brainDir = os.path.join(self.outputDir, "brain")
        # Create the output directory tree, if needed. outputDir needs
        # to be made last or else the FileExistsError will throw if the
        # outputDir already exists, leading to none of the subdirs
        # being created
        try:
            os.mkdir(self.brainDir)
            os.makedirs(self.outputDir)
        except FileExistsError:
            # If we're continuing training then this is fine
            if continueTraining:
                pass
            # Otherwise, if the directories aren't all empty, then this
            # run may be a mistake and we don't want to overwrite
            else:
                if halsey.utils.validation.is_empty_dir(self.outputDir):
                    pass
                else:
                    raise FileExistsError(
                        "Trying to peform fresh run on "
                        "a non-empty output directory. Aborting so no "
                        "overwriting occurs."
                    )
