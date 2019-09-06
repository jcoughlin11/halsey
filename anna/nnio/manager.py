"""
Title:   manager.py
Purpose: Contains the IoManager class, which is a convenience object
            for holding the Reader, Writer, and Logger objects, as well
            as various io utility functions.
Notes:
"""
import argparse

from . import logger
from . import reader
from . import writer


#============================================
#                 IoManager
#============================================
class IoManager:
    """
    A convenience object for holding a Writer, Reader, and Logger
    objects as well as various io utility functions.

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
        self.logger = logger.Logger()
        self.reader = reader.Reader()
        self.writer = writer.Writer()

    #-----
    # parse_cl_args
    #-----
    def parse_cl_args(self):
        """
        Parses the given command line arguments.

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
        # Set up the parser
        parser = argparse.ArgumentParser()
        # Parameter file
        parser.add_argument(
            "paramFile",
            help="The name of the yaml file containing parameters for the run.",
        )
        # Restart flag
        parser.add_argument(
            "--restart",
            "-r",
            action="store_true",
            help="Restarts training using the parameter file specified in the output directory.",
        )
        args = parser.parse_args()
        return args
