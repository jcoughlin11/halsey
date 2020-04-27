"""
Title: run.py
Notes:
    * call tester judge (judge.evaluate) and analyzer analyst (analyst.analyze)
"""
import gin
from rich.traceback import install as install_rich_traceback

from halsey.io.logging import log
from halsey.io.logging import setup_loggers
from halsey.io.read import parse_cl_args
from halsey.io.write import save_checkpoint
from halsey.io.write import lock_parameter_file
from halsey.utils.endrun import endrun
from halsey.utils.setup import setup_instructor


# ============================================
#                   run
# ============================================
def run():
    """
    Primary driver for training, testing, and/or analyzing a model with
    halsey.

    Parameters
    ----------
    Void

    Raises
    ------
    IOError
        Raised when the .gin file cannot be read.

    ValueError
        Raised when an unknown or invalid configurable value is
        detected.

    Returns
    -------
    Void
    """
    install_rich_traceback()
    clArgs = parse_cl_args()
    setup_loggers(clArgs)
    log("Reading parameter file...")
    try:
        gin.parse_config_file(clArgs.paramFile)
    except IOError:
        msg = f"Could not read config file: `{clArgs.paramFile}`"
        endrun(msg)
    except ValueError:
        msg = f"Unknown configurable or parameter in `{clArgs.paramFile}`."
        endrun(msg)
    lock_parameter_file(clArgs.paramFile)
    if clArgs.train:
        log("Setting up instructor...")
        instructor = setup_instructor()
        log("Training...")
        instructor.train()
        log("Saving final model...")
        save_checkpoint(instructor)
