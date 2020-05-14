"""
Title: logging.py
Notes: 
"""
import logging
import logging.handlers
import os


# ============================================
#                   log
# ============================================
def log(msg, level="info"):
    """
    Doc string.
    """
    infoFileLogger = logging.getLogger("infoFileLogger")
    infoStreamLogger = logging.getLogger("infoStreamLogger")
    if level == "info":
        infoFileLogger.info(msg)
        infoStreamLogger.info(msg)
    elif level == "warning":
        infoFileLogger.warning(msg)
        infoStreamLogger.warning(msg)
    elif level == "debug":
        infoFileLogger.debug(msg)
        infoStreamLogger.debug(msg)
    elif level == "error":
        errorLogger = logging.getLogger("errorLogger")
        infoFileLogger.error(msg)
        errorLogger.exception(msg)
        infoStreamLogger.error(msg)


# ============================================
#                setup_loggers
# ============================================
def setup_loggers():
    """
    Doc string.
    """
    # Create loggers
    loggers = initialize_loggers()
    # Logging files
    files = get_logging_files()
    # Output formaters
    formats = get_logging_formaters()
    # Handlers
    handlers = get_logging_handlers()
    # Add the formatters to the handlers
    for (handler, formatter) in zip(handlers, formatters):
        handler.setFormatter(formatter)
    # Add the handlers to the loggers
    for (logger, handler) in zip(loggers, handlers):
        logger.addHandler(handler)


# ============================================
#             initialize_loggers
# ============================================
def initialize_loggers():
    """
    Doc string.
    """
    infoStreamLogger = logging.getLogger("infoStreamLogger")
    infoFileLogger = logging.getLogger("infoFileLogger")
    errorLogger = logging.getLogger("errorLogger")
    infoStreamLogger.setLevel(logging.DEBUG)
    infoFileLogger.setLevel(logging.DEBUG)
    errorLogger.setLevel(logging.ERROR)
    return [infoStreamLogger, infoFileLogger, errorLogger]


# ============================================
#              get_logging_files 
# ============================================
def get_logging_files():
    """
    Doc string.
    """
    infoFile = os.path.join(os.getcwd(), "info.log")
    errorFile = os.path.join(os.getcwd(), "errors.log")
    return [infoFile, errorFile]


# ============================================
#             get_logging_formaters 
# ============================================
def get_logging_formaters():
    """
    Doc string.
    """
    sFmt = "Halsey - %(levelname)s - %(message)s"
    fFmt = "%(levelname)s - %(asctime)s - %(process)d - %(message)s"
    efFmt = (
        "%(levelname)s - %(asctime)s - %(process)d - (%(filename)s, "
        + "%(funcName)s, %(lineno)d) - %(message)s"
    )
    sFormatter = logging.Formatter(sFmt)
    fFormatter = logging.Formatter(fFmt, datefmt="%d-%b-%y %H:%M:%S")
    efFormatter = logging.Formatter(efFmt, datefmt="%d-%b-%y %H:%M:%S")
    return [sFormatter, fFormatter, efFormatter]


# ============================================
#            get_logging_handlers
# ============================================
def get_logging_handlers():
    """
    Doc string.
    """
    sHandler = logging.StreamHandler()
    fHandler = logging.handlers.RotatingFileHandler(
        infoFile, maxBytes=250000000, backupCount=5, delay=True
    )
    efHandler = logging.handlers.RotatingFileHandler(
        errorFile, maxBytes=250000000, backupCount=5, delay=True
    )
    sHandler.setLevel(logging.DEBUG)
    fHandler.setLevel(logging.INFO)
    efHandler.setLevel(logging.ERROR)
    # Colorize, if desired
    sHandler.emit = colorize_logging(sHandler.emit)
    return [sHandler, fHandler, eHandler]


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
