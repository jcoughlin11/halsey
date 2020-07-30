"""
Title: run.py
Notes:
"""
from halsey.utils.setup import initialize
from halsey.utils.setup import setup_instructor
from halsey.utils.setup import setup_proctor


# ============================================
#                    run
# ============================================
def run():
    """
    Primary driver program for running `halsey` from the command line.

    `halsey` has three main modes of operation: training, evaluating,
    and analyzing a model. This function oversees the initialization
    of a run as well as each of the three operations.
    """
    params, clArgs = initialize()
    if clArgs.train:
        instructor = setup_instructor()
        instructor.train()
    if clArgs.evaluate:
        proctor = setup_proctor()
        proctor.evaluate()
