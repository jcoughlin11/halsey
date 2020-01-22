"""
Title: env.py
Purpose: Creates and manages the game environment.
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
    Creates a new game environment.

    Parameters
    ----------
    envName : str
        The name of the game environment to build.

    Raises
    ------
    None

    Returns
    -------
    env : gym.Env
        The fully constructed game environment.
    """
    env = gym.make(envName)
    return env
