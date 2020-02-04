"""
Title: endrun.py
Purpose: Contains functions related stopping and cleanly exiting.
Notes:
"""
import os


# ============================================
#             check_early_stop
# ============================================
def check_early_stop():
    """
    Checks the two conditions for user-instigated early stopping: the
    presence of a file called 'stop' and whether or not the run time
    has surpassed the given time limit.

    Parameters
    ----------
    None

    Raises
    ------
    None

    Returns
    -------
    earlyStop : bool
        If True, the code should be terminated immediately.
    """
    earlyStop = False
    if os.path.isfile(os.path.join(os.getcwd(), "stop")):
        earlyStop = True
        os.remove(os.path.join(os.getcwd(), "stop"))
    return earlyStop
