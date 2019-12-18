"""
Title: env.py
Purpose:
Notes:
    * The environment handles interaction between the game and the
        Agent
"""
import gym


# ============================================
#                 build_env
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
