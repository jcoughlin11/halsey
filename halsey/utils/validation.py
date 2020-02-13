"""
Title: validation.py
Purpose:
Notes:
"""
import logging
import os
import sys

import halsey


# ============================================
#                registers
# ============================================

# -----
# paramRegister
# -----
paramRegister = {
    "envName": str,
    "train": bool,
    "test": bool,
    "timeLimit": int,
    "outputDir": str,
    "fileBase": str,
    "nEpisodes": int,
    "maxEpisodeSteps": int,
    "batchSize": int,
    "savePeriod": int,
    "architecture": str,
    "discount": float,
    "learningRate": float,
    "loss": str,
    "optimizer": str,
    "fixedQSteps": int,
    "maxSize": int,
    "pretrainLen": int,
    "epsDecayRate": float,
    "epsilonStart": float,
    "epsilonStop": float,
    "cropBot": int,
    "cropLeft": int,
    "cropRight": int,
    "cropTop": int,
    "shrinkRows": int,
    "shrinkRows": int,
    "traceLen": int,
}


# -----
# optionRegister
# -----
optionRegister = {}


# -----
# rnnRegister
# -----
rnnRegister = []


# -----
# lossRegister
# -----
lossRegister = {}


# -----
# optimizerRegister
# -----
optimizerRegister = {}


# ============================================
#              register_option
# ============================================
def register_option(cls):
    """
    This wrapper takes in an object used to represent an option and
    logs it for easier validation and object creation.

    Parameters
    ----------
    cls : Object
        The option's class.

    Raises
    ------
    None

    Returns
    -------
    cls : Object
        The unaltered passed-in object.
    """
    optionRegister[cls.__name__] = cls
    return cls


# ============================================
#              register_network
# ============================================
def register_network(cls):
    """
    This wrapper takes in a network name and logs it for easier
    network creation.

    Parameters
    ----------
    cls : tf.keras.Model
        The network's class.

    Raises
    ------
    None

    Returns
    -------
    cls : tf.keras.Model
        The unaltered passed-in object.
    """
    optionRegister[cls.__name__] = cls
    # The rnn register is needed for setting the channelsFirst variable
    # that ends up determining the format of the network's input as
    # eitehr NCHW or NHWC
    try:
        if cls.networkType == "rnn":
            rnnRegister.append(cls.__name__)
    except AttributeError:
        # At this point, the loggers haven't been configured yet, so we
        # do it here
        mockArgs = halsey.utils.folio.Folio()
        setattr(mockArgs, "silent", False)
        setattr(mockArgs, "noColoredLogs", False)
        halsey.utils.logger.configure_loggers(mockArgs)
        msg = f"Network `{cls.__name__}` does not have a networkType attribute."
        infoLogger = logging.getLogger("infoLogger")
        errorLogger = logging.getLogger("errorLogger")
        infoLogger.error(msg)
        errorLogger.exception(msg)
        sys.exit(1)
    return cls


# ============================================
#               register_loss
# ============================================
def register_loss(func):
    """
    Adds the given loss function to halsey's list of known losses.

    This list should only contain custom loss functions that are
    not already known to keras.

    The name of the loss function should match what's passed in for the
    loss function in the parameter file.

    Parameters
    ----------
    func : function
        The custom loss function to be registered.

    Raises
    ------
    None

    Returns
    -------
    func : function
        The unaltered custom loss function.
    """
    lossRegister[func.__name__] = func
    return func


# ============================================
#            register_optimizer
# ============================================
def register_optimizer(cls):
    """
    Adds the given optimizer class to halsey's list of known
    optimizers.

    This list should only contain custom optimizers that are not
    already known to keras.

    The name of the optimizer class should match what's passed in for
    the optimizer in the parameter file.

    Parameters
    ----------
    cls : tf.keras.optimizers.Optimizer
        The custom optimizer class. Must be a subclass of
        `tf.keras.optimizers.Optimizer`.

    Raises
    ------
    None

    Returns
    -------
    cls : tf.keras.optimizers.Optimizer
        The now-registered custom optimizer class.
    """
    optimizerRegister[cls.__name__] = cls
    return cls


# ============================================
#              validate_params
# ============================================
def validate_params(params):
    """
    Makes sure that every passed parameter is both known to Halsey as
    well as the correct type.

    Parameters
    -----------
    params : dict
        The raw parameters as read in from the parameter file.

    Raises
    -------
    None

    Returns
    -------
    None
    """
    for section, sectionParams in params.items():
        for paramName, paramVal in sectionParams.items():
            # See if the parameter is in paramRegister
            if paramName in paramRegister:
                try:
                    assert isinstance(paramVal, paramRegister[paramName])
                except AssertionError:
                    infoLogger = logging.getLogger("infoLogger")
                    errorLogger = logging.getLogger("errorLogger")
                    msg = (
                        f"Parameter `{paramName}` is not instance "
                        + f"of `{paramRegister[paramName]}`."
                    )
                    infoLogger.error(msg)
                    errorLogger.exception(msg)
                    sys.exit(1)
            # If it's not, it might be an option
            elif paramName in optionRegister:
                continue
            else:
                infoLogger = logging.getLogger("infoLogger")
                errorLogger = logging.getLogger("errorLogger")
                msg = "Unrecognized parameter `{paramName}`."
                infoLogger.error(msg)
                errorLogger.error(msg)
                sys.exit(1)


# ============================================
#                is_empty_dir
# ============================================
def is_empty_dir(directory):
    """
    Checks to see if the output directory tree is empty.

    This is to prevent overwriting a directory already in use by
    another run.

    Parameters
    ----------
    directory : str
        The root of the output directory tree.

    Raises
    ------
    None

    Returns
    -------
    isEmpty : bool
        If True then the directory tree is empty and we can proceed.
    """
    isEmpty = True
    # Walk the directory tree
    for root, dirs, files in os.walk(directory):
        # If there are files, then we're done
        if len(files) != 0:
            isEmpty = False
            break
    return isEmpty
