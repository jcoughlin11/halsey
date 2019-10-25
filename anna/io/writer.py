"""
Title:   writer.py
Purpose: Contains the writer class
Notes:
"""
import os

import yaml


# ============================================
#                  Writer
# ============================================
class Writer:
    """
    Used to write data to files.

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
        self.baseDir = None
        self.fileBase = None

    # -----
    # set_params
    # -----
    def set_params(self, params):
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
        self.baseDir = params.outputDir
        self.fileBase = params.fileBase

    # -----
    # save_checkpoint
    # -----
    def save_checkpoint(self, brain, memory, navigator, trainer):
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
        # Make the checkpoints directory, if needed
        dirName = os.path.join(self.baseDir, "checkpoints")
        if not os.path.isdir(dirName):
            os.makedirs(dirName)
        # Save the brain
        self.save_brain(dirName, brain)
        # Save the navigator
        self.save_navigator(dirName, navigator)
        # Save the trainer
        self.save_trainer(dirName, trainer)
        # Save the memory
        self.save_memory(dirName, memory)

    # -----
    # save_brain
    # -----
    def save_brain(self, dirName, brain):
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

    # -----
    # save_param_file
    # -----
    def save_param_file(self, relay):
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
        fn = os.path.join(self.baseDir, self.fileBase + "_params_backup.yaml")
        with open(fn, "w") as f:
            yaml.dump(f)
