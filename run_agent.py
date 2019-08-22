"""
Title:   run_agent.py
Author:  Jared Coughlin
Date:    3/19/19
Purpose: Driver code for using DQL to train an agent to play a game
Notes:
"""
import sys

import anna


# ============================================
#                   main
# ============================================
def main():
    """
    Driver for training or running an agent instance.

    Parameters:
    -----------
        None

    Raises:
    -------
        None

    Returns:
    --------
        None
    """
    # Do an args check
    try:
        paramFile = sys.argv[1]
    except IndexError:
        print("Error, must pass a parameter file!")
        sys.exit(1)
    try:
        if int(sys.argv[2]) == 1:
            restartFlag = True
        else:
            restartFlag = False
    except ValueError:
        print("Error, restart flag must be an integer!")
        sys.exit(1)
    except IndexError:
        restartFlag = False
    # Initialize the run
    agent = anna.agents.qagent.Agent(paramFile, restartFlag)
    # Train
    agent.train()
    # Test
    agent.test()


# ============================================
#                main script
# ============================================
if __name__ == "__main__":
    main()
