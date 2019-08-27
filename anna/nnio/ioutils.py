"""
Title:   ioutils.py
Author:  Jared Coughlin
Date:    8/27/19
Purpose: Contains miscellaneous I/O functions
Notes:
"""
import argparse


#============================================
#                 parse_args
#============================================
def parse_args():
    """
    Sets up and collects the given command line arguments. These are:

        paramFile : string : required
            The name of the parameter file to read.

        --restart, -r : string : optional
            Flag indicating to restart training from the beginning
            using the given parameter file. If this is not present then
            the behavior defaults to looking for a restart file with
            which to continue training. If no restart file is found,
            the code begins a new training session with the given
            parameter file.

    Parameters:
    -----------
        None

    Raises:
    -------
        None

    Returns:
    --------
        args : argparse.Namespace
            A class whose attributes are the names of the known args
            given by calls to add_argument.
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
        help="Restarts training using the given parameter file.",
    )
    args = parser.parse_args()
    return args
