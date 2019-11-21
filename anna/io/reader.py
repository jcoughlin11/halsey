"""
Title:   reader.py
Purpose: Contains the Reader class.
Notes:
"""
import argparse
import glob
import os

import yaml


# ============================================
#                  Reader
# ============================================
class Reader:
    """
    Used to read data from files.

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
    # parse_cl_args
    # -----
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
        # Parameter file (if -- is not a prefix to the option name then
        # argparse assumes it's a positional argument and therefore
        # required by default
        parser.add_argument(
            "paramFile",
            help="The name of the yaml file containing parameters for the run.",
        )
        # Training continuation flag
        parser.add_argument(
            "--continue",
            "-c",
            dest="continueTraining",
            action="store_true",
            default=False,
            help="Continues training with parameter file in output directory.",
        )
        args = parser.parse_args()
        return args

    # -----
    # read_param_file
    # -----
    def read_param_file(self, paramFile, continueTraining):
        """
        Reads in the parameters from the given parameter file. See the
        README for a list and description of each parameter.

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
        # Read the file
        paramFile = os.path.abspath(os.path.expanduser(paramFile))
        with open(paramFile, "r") as f:
            params = yaml.load(f, Loader=yaml.Loader)
        # If continuing training, read the saved copy of the original
        # parameter file. This guards against changes made to passed-in
        # version since the original run
        if continueTraining:
            outDir = os.path.abspath(
                os.path.expanduser(params["io"]["outputDir"])
            )
            paramFiles = glob.glob(os.path.join(outDir, "*_backup.yaml"))
            with open(paramFiles[0], "r") as f:
                params = yaml.load(f)
        return params

    # -----
    # set_params
    # -----
    def set_params(self, ioParams):
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
        self.baseDir = ioParams.outputDir
        self.fileBase = ioParams.fileBase
