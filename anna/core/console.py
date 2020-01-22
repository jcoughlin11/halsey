"""
Title:   console.py
Purpose: Primary driver for using ANNA from the command line
Notes:
"""
import sys

import anna


# ============================================
#                     run
# ============================================
def run():
    """
    Driver function for using anna.

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
    # Set up the agent
    agent = anna.Agent()
    # Train
    if agent.trainingEnabled:
        if not agent.train():
            sys.exit()
