"""
Title: utils.py
Purpose:
Notes:
    * The navigator class oversees the environment, frame processing,
        and action selection
    * The navigator class handles transitioning from state to state
"""
import anna

from .framebyframe import FrameByFrameNavigator


# ============================================
#            get_new_navigator
# ============================================
def get_new_navigator(navParams, actionParams, frameParams, envName):
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
    actionManager = anna.actions.utils.get_new_action_manager(actionParams)
    # Build the navigator
    if navParams.mode == "frameByFrame":
        navigator = FrameByFrameNavigator(
            navParams, env, frameManager, actionManager
        )
    return navigator
