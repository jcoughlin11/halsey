"""
Title:   reader.py
Purpose: Contains the Reader class.
Notes:
"""
import argparse
import glob
import os

import yaml


#============================================
#                  Reader
#============================================
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
        self.baseDir = None
        self.fileBase = None

    #-----
    # set_params
    #-----
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

    #-----
    # read_param_file
    #-----
    def read_param_file(self, paramFile):
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
        # If we're given a dir, as in the case of continuing training,
        # look for a yaml file to load
        if os.path.isdir(paramFile):
            paramFiles = glob.glob(os.path.join(paramFile, '*_backup.yaml'))
            if len(paramFiles) != 1:
                msg = "Couldn't determine backup parameter file in output " \
                    "directory: {}.".format(paramFile)
                raise FileNotFoundError(msg)
            paramFile = paramFiles[0]
        # Read the file
        paramFile = os.path.abspath(os.path.expanduser(paramFile))
        with open(paramFile, 'r') as f:
            params = yaml.load(f)
        return params

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
            required=True,
            help="The name of the yaml file containing parameters for the run.",
        )
        # Training continuation flag
        parser.add_argument(
            "--continue",
            "-c",
            dest='continueTraining',
            action="store_true",
            help="Continues training with parameter file in output directory.",
        )
        args = parser.parse_args()
        return args
