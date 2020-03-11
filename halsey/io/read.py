"""
Title: read.py
Notes:
"""
import argparse


# ============================================
#               parse_cl_args
# ============================================
def parse_cl_args():
    """
    Handles parsing and setting of arguments passed to halsey via the
    command-line.

    Available options are:

        paramFile : required
            The name of the .gin file containing the parameters to be
            used for the current run.

        silent : optional
            If set, then logging to stdout is suppressed. Log files
            will still be written, however.

        no-color : optional
            If set, then all logging is done without colored text.

        no-train : optional
            If set, then training will be skipped for the current run.
            Otherwise, training will occur.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paramFile",
        help="The name of the .gin file containing parameters for the run.",
    )
    parser.add_argument(
        "--silent",
        "-s",
        dest="silent",
        action="store_true",
        help="Suppress writing to stdout. Log files are still written.",
    )
    parser.add_argument(
        "--no-color",
        "-nc",
        dest="noColor",
        action="store_true",
        help="If set, colored text is disabled for all output and logs.",
    )
    parser.add_argument(
        "--no-train",
        "-nt",
        dest="train",
        action="store_false",
        help="If set, skip the training step.",
    )
    args = parser.parse_args()
    return args
