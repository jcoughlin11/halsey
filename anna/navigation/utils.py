"""
Title: utils.py
Purpose: Contains helper functions relating to setting up and going
            through the game world.
Notes:
"""


# ============================================
#             get_new_navigator
# ============================================
def get_new_navigator(envName, navParams, exploreParams, frameParams):
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
    env = anna.utils.env.build_env(envName)
    # Build the frame manager
    frameManager = anna.frames.utils.get_new_frame_manager(frameParams)
    # Build the action manager
    actionManager = anna.actions.utils.get_new_action_manager(exploreParams)
    # Build the navigator
    if navParams.mode == "frameByFrame":
        navigator = FrameByFrameNavigator(
            env, navParams, frameManager, actionManager
        )
    return navigator
