"""
Title:      endrun.py
Purpose:    Contains functions related stopping and cleanly exiting.
Notes:
"""
import logging
import os
import sys


# ============================================
#                   endrun
# ============================================
def endrun(exception, msg):
    """
    Boilerplate for logging an exception's traceback as well as the
    code's error message for why the exception occurred.

    Paremters
    ---------
    Void

    Raises
    ------
    Void

    Returns
    -------
    Void
    """
    infoLogger = logging.getLogger("infoLogger")
    errorLogger = logging.getLogger("errorLogger")
    infoLogger.error(msg)
    errorLogger.exception(msg)
    sys.exit()


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
    Void

    Raises
    ------
    Void

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
