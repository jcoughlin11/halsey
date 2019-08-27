"""
Title:   run_agent.py
Author:  Jared Coughlin
Date:    3/19/19
Purpose: Driver code for using DQL to train an agent to play a game
Notes:
"""
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
    # Get args from the command line
    args = anna.nnio.ioutils.parse_args()
    # Set up the agent 
    agent = anna.agents.qagent.Agent(args)
    # Train
    agent.train()
    # Test
    agent.test()


# ============================================
#                main script
# ============================================
if __name__ == "__main__":
    main()
