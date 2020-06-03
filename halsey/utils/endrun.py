"""
Title: endrun.py
Notes:
"""
import sys

from halsey.io.logging import log


# ============================================
#                   endrun
# ============================================
def endrun(msg, level="info"):
    """
    Boilerplate for logging an exception's traceback as well as the
    code's error message for why the exception occurred.

    Paremters
    ---------
    msg : str
        The error message.

    Raises
    ------
    Void

    Returns
    -------
    Void
    """
    log(msg, level)
    sys.exit()
