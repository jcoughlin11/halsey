"""
Title: env.py
Purpose: Contains utility functions related to the game environment.
Notes:
"""
import gym


# ============================================
#                build_env
# ============================================
def build_env(envName):
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
    env = gym.make(envName)
    return env
