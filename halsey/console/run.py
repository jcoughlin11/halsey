"""
Title:      run.py
Purpose:    Primary driver for using halsey from the command line
Notes:
"""
from halsey.utils.setup import setup


# ============================================
#                     run
# ============================================
def run():
    """
    Driver function for using halsey as a command-line tool.

    Using halsey from the command line is meant to be end-to-end. That
    is, the configuration file is read in and parsed and then a model
    is created, trained, tested, and/or analyzed.

    Parameters
    ----------
    Void

    Raises
    ------
    Void

    Returns
    -------
    Void
    """
    agent = setup()
    if agent.trainingEnabled:
        agent.train()
