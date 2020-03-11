"""
Title: endrun.py
Notes:
"""
import logging
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
    exception : python.Exception
        The error being raised.

    msg : str
        The error message.

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
