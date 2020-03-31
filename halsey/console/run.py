"""
Title: run.py
Notes:
    * call tester judge (judge.evaluate) and analyzer analyst (analyst.analyze)
"""
import gin

from halsey.io.logging import setup_loggers
from halsey.io.read import parse_cl_args
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
    clArgs = parse_cl_args()
    setup_loggers(clArgs)
    try:
        gin.parse_config_file(clArgs.paramFile)
    except IOError:
        msg = f"Could not read config file: `{clArgs.paramFile}`"
        endrun(msg)
    except ValueError:
        msg = f"Unknown configurable or parameter in `{clArgs.paramFile}`."
        endrun(msg)
    if clArgs.train:
        instructor = setup_instructor()
        instructor.train()
