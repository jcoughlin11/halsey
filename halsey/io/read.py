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
    Doc string.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paramFile",
        help="The name of the .gin file containing parameters for the run.",
    )
    args = parser.parse_args()
    return args
