"""
Title: logging.py

Notes: * The work done in setup_loggers is not done at the module
         level because it depends on the command-line arguments that
         are passed.
"""
import logging
import logging.handlers
import os


# ============================================
#            colorize_logging
# ============================================
def colorize_logging(emit_func):
    """
    Adds color to the logs emitted by the given handler's emit()
    method.

    .. note::

        This function is from the
        `yt project <https://github.com/yt-project/yt>`_ which, in
        turn, is based on `this method <https://tinyurl.com/kd4gbxf>`_.

    Parameters
    ----------
    emit_func : logging.Handler().emit()
        The method responsible for outputting the log information.

    Raises
    ------
    Void

    Returns
    -------
    Void
    """

    def colorize(*args):
        levelno = args[0].levelno
        if levelno >= 50:
            color = "\x1b[31m"  # red
        elif levelno >= 40:
            color = "\x1b[31m"  # red
        elif levelno >= 30:
            color = "\x1b[33m"  # yellow
        elif levelno >= 20:
            color = "\x1b[32m"  # green
        elif levelno >= 10:
            color = "\x1b[35m"  # pink
        else:
            color = None
        if color is not None:
            ln = color + args[0].levelname + "\x1b[0m"
        else:
            ln = args[0].levelname
        args[0].levelname = ln
        return emit_func(*args)

    return colorize


# ============================================
#                setup_loggers
# ============================================
def setup_loggers(clArgs):
    """
    Sets up the formats, streams, and handlers for the loggers.

    The logging is separated into two different loggers: one to manage
    normal code output (called infoLogger) and the other to exclusively
    handle errors (called errorLogger).

    The information handled by infoLogger is written to both stdout and
    an info.log file by default. If the --silent option has been passed
    then infoLogger will only output to the info.log file.

    The information handled by errorLogger is written to a file called
    errors.log. This file contains both the full traceback information
    as well as an info line containing both a helpful (hopefully) error
    message as well as the exact location of the error in the code.
    This is meant to be more readily readable than the full traceback.

    By default, all of the logging information is output in colored
    text. This can be disabled by passing the --no-color option on the
    command-line.

    The location of the log files is the current working directory.

    This separation was done to keep stdout cleaner as well as to
    facilitate easier locating of the info you care about, such as
    training status or where things went wrong.

    Parameters
    ----------
    clArgs : argparse.Namespace
        The command-line arguments passed to Halsey. Used to determine
        log behavior.

    Raises
    ------
    Void

    Returns
    -------
    Void
    """
    # Create loggers
    infoLogger = logging.getLogger("infoLogger")
    errorLogger = logging.getLogger("errorLogger")
    # Logging files
    infoFile = os.path.join(os.getcwd(), "info.log")
    errorFile = os.path.join(os.getcwd(), "errors.log")
    # Output formats
    sFmt = "Halsey - %(levelname)s - %(message)s"
    fFmt = "%(levelname)s - %(asctime)s - %(process)d - %(message)s"
    efFmt = (
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
    efFormatter = logging.Formatter(efFmt, datefmt="%d-%b-%y %H:%M:%S")
    # Set levels
    infoLogger.setLevel(logging.DEBUG)
    errorLogger.setLevel(logging.ERROR)
    sHandler.setLevel(logging.DEBUG)
    fHandler.setLevel(logging.INFO)
    efHandler.setLevel(logging.ERROR)
    # Add the formatters to the handlers
    sHandler.setFormatter(sFormatter)
    fHandler.setFormatter(fFormatter)
    efHandler.setFormatter(efFormatter)
    # Add the handlers to the loggers
    infoLogger.addHandler(fHandler)
    errorLogger.addHandler(efHandler)
    # Colorize, if desired
    if not clArgs.noColor:
        sHandler.emit = colorize_logging(sHandler.emit)
    infoLogger.addHandler(sHandler)


# ============================================
#                   log
# ============================================
def log(msg, level="info"):
    """
    Abstracts away the boilerplate code for logging a message.

    Parameters
    ----------
    msg : str
        The message to log.

    level : str
        The logging level (debug, info, warning, or error).

    Raises
    ------
    ValueError
        Raised when either no message or an invalid level is passed.

    Returns
    -------
    Void
    """
    infoLogger = logging.getLogger("infoLogger")
    if level == "info":
        infoLogger.info(msg)
    elif level == "warning":
        infoLogger.warning(msg)
    elif level == "debug":
        infoLogger.debug(msg)
    elif level == "error":
        errorLogger = logging.getLogger("errorLogger")
        infoLogger.error(msg)
        errorLogger.exception(msg)
