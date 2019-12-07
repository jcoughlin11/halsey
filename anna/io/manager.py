"""
Title:   manager.py
Purpose:
Notes:
"""
import os

import anna


# ============================================
#                 IoManager
# ============================================
class IoManager:
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
        self.reader = anna.io.reader.Reader()
        self.writer = anna.io.writer.Writer()
        self.fileBase = None
        self.outputDir = None
        self.memoryDir = None
        self.envDir = None
        self.brainDir = None
        self.trainerDir = None

    # -----
    # load_params
    # -----
    def load_params(self):
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
        # Parse the command-line args
        clArgs = self.reader.parse_args()
        # Read the parameter file
        params = self.reader.read_param_file(
            clArgs.paramFile, clArgs.continueTraining
        )
        # Build the folio
        folio = anna.utils.folio.get_new_folio(clArgs, params)
        # Validate the parameters
        anna.utils.validation.validate_params(folio)
        # Set the io parameters
        self.set_io_params(folio.io)
        return folio

    # -----
    # save_params
    # -----
    def save_params(self, folio):
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
        self.writer.save_params(folio, self.outputDir)

    # -----
    # load_checkpoint
    # -----
    def load_checkpoint(self):
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

    # -----
    # save_checkpoint
    # -----
    def save_checkpoint(self):
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

    # -----
    # load_brain
    # -----
    def load_brain(self):
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

    # -----
    # set_io_params
    # -----
    def set_io_params(self, ioParams):
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
        # Set the names of the various output directories
        self.fileBase = ioParams.fileBase
        self.outputDir = os.path.abspath(os.path.expanduser(ioParams.outputDir))
        self.memoryDir = os.path.join(self.outputDir, "memory")
        self.envDir = os.path.join(self.outputDir, "environment")
        self.brainDir = os.path.join(self.outputDir, "brain")
        self.trainerDir = os.path.join(self.outputDir, "trainer")
        # Create the output directory tree, if needed
        os.makedirs(self.outputDir)
        os.mkdir(self.memoryDir)
        os.mkdir(self.envDir)
        os.mkdir(self.brainDir)
        os.mkdir(self.trainerDir)
