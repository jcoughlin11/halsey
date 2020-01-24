"""
Title:   console.py
Purpose: Primary driver for using Halsey from the command line
Notes:
"""
import sys

import halsey


# ============================================
#                     run
# ============================================
def run():
    """
    Driver function for using halsey.

    Instantiates an instance of the Agent class and then either
    trains the agent, tests the agent, or both.

    Parameters
    ----------
    None

    Raises
    ------
    None

    Returns
    -------
    None
    """
    agent = halsey.Agent()
    agent.setup()
    if agent.trainingEnabled:
        if not agent.train():
            sys.exit()
