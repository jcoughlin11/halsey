"""
Title:      logger.py
Purpose:    Contains the logger class for custom tracking of exceptions
                and code information.
Notes:
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

    Raises
    ------
    Void

    Returns
    -------
    colorize : function
        If using colored logs, this will be the modified emit function
        that includes the color codes. Otherwise, it will be the
        original emit function.
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
#            configure_loggers
# ============================================
def configure_loggers(silent, noColoredLogs):
    """
    Sets up the formatting, streams, and handlers for the loggers.

    The logging is separated into two different loggers rather than
    one logger with two handlers in order to keep each log file
    cleaner and not clutter up stdout.

    Parameters
    ----------
    silent : bool
        Command-line option. If set, then logging to stdout is
        disabled.

    noColoredLogs : bool
        Command-line option. If set, then the logs are output without
        color.

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
    fFmt = (
        "%(levelname)s - %(asctime)s - %(process)d - (%(filename)s, "
        + "%(funcName)s, %(lineno)d) - %(message)s"
    )
    # Handlers
    if not silent:
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
    if not silent:
        sHandler.setLevel(logging.DEBUG)
    fHandler.setLevel(logging.INFO)
    efHandler.setLevel(logging.ERROR)
    # Add the formatters to the handlers
    if not silent:
        sHandler.setFormatter(sFormatter)
    fHandler.setFormatter(fFormatter)
    efHandler.setFormatter(fFormatter)
    # Add the handlers to the loggers
    infoLogger.addHandler(fHandler)
    errorLogger.addHandler(efHandler)
    if not silent:
        # Colorize, if desired
        if not noColoredLogs:
            sHandler.emit = colorize_logging(sHandler.emit)
        infoLogger.addHandler(sHandler)
