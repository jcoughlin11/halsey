"""
Title:   run_anna.py
Purpose: Primary driver for using ANNA
Notes:
"""
import anna


#============================================
#                   main
#============================================
def main():
    """
    Creates an instance of the Agent object and then trains and tests
    it.

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
    agent = anna.agents.qagent.Agent()
    # Train, if applicable
    if agent.trainingEnabled:
        agent.train()
    # Test, if applicable
    if agent.testingEnabled:
        agent.test()


#============================================
#               main script
#============================================
if __name__ == '__main__':
    main()
