"""
Title: reader.py
Purpose:
Notes:
"""
import argparse
import os
import sys

import yaml


# ============================================
#                  Reader
# ============================================
class Reader:
    """
    Handles the reading of all files and the parsing of all
    command-line arguments.

    Attributes
    ----------
    None

    Methods
    -------
    parse_args()
        Parses the command-line arguments.

    read_param_file(paramFile, continueTraining)
        Reads in the given parameter file.
    """

    # -----
    # parse_cl_args
    # -----
    def parse_args(self):
        """
        Parses the given command-line arguments.

        Parameters
        ----------
        None

        Raises
        ------
        None

        Returns
        -------
        args : argparse.Namespace
            Object for holding the parsed command-line arguments.
        """
        # Set up the parser
        parser = argparse.ArgumentParser()
        # Parameter file (if -- is not a prefix to the option name then
        # argparse assumes it's a positional argument and therefore
        # required by default)
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
    def read_param_file(self, paramFile):
        """
        Reads the given parameter file.

        Parameters
        ----------
        paramFile : str
            The base name of the parameter file to read.

        Raises
        ------
        None

        Returns
        -------
        params : dict
            A nested dictionary containing the data read in from the
            parameter file. Has the same structure as the parameter
            file itself.
        """
        # Read the file
        paramFile = os.path.abspath(os.path.expanduser(paramFile))
        try:
            with open(paramFile, "r") as f:
                params = yaml.safe_load(f)
        except IOError:
            msg = "Error: Can't open parameter file: `{}` for reading.".format(
                paramFile
            )
            print(msg)
            sys.exit(1)
        return params
