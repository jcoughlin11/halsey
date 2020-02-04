"""
Title: logger.py
Purpose: Contains the logger class for custom tracking of exceptions
            and code information.
Notes:
"""
import logging
import logging.handlers
import os


def configure_loggers():
    """
    Sets up the formatting, streams, and handlers for the loggers.

    The logging is separated into two different loggers rather than
    one logger with two handlers in order to keep each log file
    cleaner and not clutter up stdout.

    Parameters
    ----------
    pass

    Raises
    ------
    pass

    Returns
    -------
    pass
    """
    # Create loggers
    infoLogger = logging.getLogger("infoLogger")
    errorLogger = logging.getLogger("errorLogger")
    # Logging files
    infoFile = os.path.join(os.getcwd(), "info.log")
    errorFile = os.path.join(os.getcwd(), "errors.log")
    # Output formats
    sFmt = "Halsey - %(levelname)s - %(message)s"
    fFmt = (
        "%(levelname)s - %(asctime)s - %(process)d - (%(filename)s, "
        + "%(funcName)s, %(lineno)d) - %(message)s"
    )
    # Handlers
    sHandler = logging.StreamHandler()
    fHandler = logging.handlers.RotatingFileHandler(
        infoFile, maxBytes=250000000, backupCount=5, delay=True
    )
    efHandler = logging.handlers.RotatingFileHandler(
        errorFile, maxBytes=250000000, backupCount=5, delay=True
    )
    # Formatters
    sFormatter = logging.Formatter(sFmt)
    fFormatter = logging.Formatter(fFmt, datefmt="%d-%b-%y %H:%M:%S")
    # Set levels
    infoLogger.setLevel(logging.DEBUG)
    errorLogger.setLevel(logging.ERROR)
    sHandler.setLevel(logging.DEBUG)
    fHandler.setLevel(logging.INFO)
    efHandler.setLevel(logging.ERROR)
    # Add the formatters to the handlers
    sHandler.setFormatter(sFormatter)
    fHandler.setFormatter(fFormatter)
    efHandler.setFormatter(fFormatter)
    # Add the handlers to the loggers
    infoLogger.addHandler(sHandler)
    infoLogger.addHandler(fHandler)
    errorLogger.addHandler(efHandler)
