"""
Title:      read.py
Purpose:    Contains functions for reading data from the command line
                and from files.
Notes:
"""
import argparse


# ============================================
#               parse_cl_args
# ============================================
def parse_cl_args():
    """
    Parses the given command-line arguments.

    Parameters
    ----------
    Void

    Raises
    ------
    Void

    Returns
    -------
    args : argparse.Namespace
        Object for holding the parsed command-line arguments.
    """
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
        help="Continues training with parameter file in output directory.",
    )
    # Verbose output
    parser.add_argument(
        "--silent",
        "-s",
        dest="silent",
        action="store_true",
        help="Suppress writing to stdout. Log files are still written.",
    )
    # Colored log output
    parser.add_argument(
        "--no-color",
        "-nc",
        dest="noColoredLogs",
        action="store_true",
        help="If set, then the logs sent to std out will not be colored",
    )
    args = parser.parse_args()
    return args
