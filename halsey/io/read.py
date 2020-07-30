"""
Title: read.py
Notes:
"""
import argparse


# ============================================
#                parse_cl_args
# ============================================
def parse_cl_args():
    """
    Handles registration and parsing of command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paramFile",
        help="The name of the .gin file containing parameters for the run.",
    )
    parser.add_argument(
        "--train",
        "-t",
        dest="train",
        action="store_true",
        help="Instructs the code to train a model.",
    )
    parser.add_argument(
        "--evaluate",
        "-e",
        dest="evaluate",
        action="store_true",
        help="Instructs the code to evaluate a model.",
    )
    clArgs = parser.parse_args()
    return clArgs
