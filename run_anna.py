"""
Title:   run_anna.py
Purpose: Primary driver for using ANNA
Notes:
"""
import sys

import anna


# ============================================
#                   main
# ============================================
def main():
    """
    Doc string.

    Parameters:
    -----------
        pass

    Raises:
    -------
        pass

    Returns:
    --------
        pass
    """
    # Set up the agent
    agent = anna.agent.Agent()
    # Train
    if agent.trainingEnabled:
        if not agent.train():
            sys.exit()
    # Test
    if agent.testingEnabled:
        agent.test()


# ============================================
#               main script
# ============================================
if __name__ == "__main__":
    main()
