"""
Title: utils.py
Purpose: Contains helper functions relating to setting up and going
            through the game world.
Notes:
"""


#============================================
#             get_new_navigator
#============================================
def get_new_navigator(navParams, explorer, frameHandler, env):
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
    # Build the gym environment
    if navParams.mode == 'frameByFrame':
        navigator = FrameByFrameNavigator(navParams, explorer, frameHandler, env)
    return navigator


#============================================
#                 build_env
#============================================
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
    # Build the named environment
    env = gym.make(envName)
    # Call reset so that everything works later on (seed setting,
    # loading the emulator state, etc.)
    state = env.reset()
    return env
