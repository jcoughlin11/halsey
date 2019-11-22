"""
Title: endrun.py
Purpose: Contains functions related to clean-up and early stopping.
Notes:
"""
import os
import time


# ============================================
#             check_early_stop
# ============================================
def check_early_stop(startTime, timeLimit):
    """
    Doc string. Early stopping happens for one of two reasons: a stop
    file is detected or the user-defined time-limit is reached.

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
    earlyStop = False
    stopFile = os.path.join(os.getcwd(), "stop")
    curTime = time.time()
    # Check for existence of stop file
    if os.path.isfile(stopFile):
        earlyStop = True
        os.remove(stopFile)
    # Check for time-limit reached
    if curTime - startTime > timeLimit:
        earlyStop = True
    return earlyStop
