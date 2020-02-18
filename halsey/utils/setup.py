"""
Title:      setup.py
Purpose:    Contains functions for initializing a run.
Notes:
"""
import gin

from halsey.io.logger import setup_loggers
from halsey.io.read import parse_cl_args

from .endrun import endrun
from .object_management import get_agent


# ============================================
#                   setup
# ============================================
def setup():
    """
    Primary driver function for initializing a run.

    Handles parsing the command-line arguments, setting up the loggers,
    reading in the gin configuration file,  and then instantiating an
    agent.

    Parameters
    ----------
    Void

    Raises
    ------
    IOError
        If the gin configuration file cannot be read.

    Returns
    -------
    agent : halsey.agents.base.BaseAgent
        The object doing the learning, being tested, and/or being
        analyzed.
    """
    clArgs = parse_cl_args()
    setup_loggers(clArgs.silent, clArgs.noColoredLogs)
    try:
        gin.parse_config_file(clArgs.paramFile)
    except IOError as e:
        msg = f"Could not read config file: `{clArgs.paramFile}`"
        endrun(e, msg)
    agent = get_agent()
    return agent
