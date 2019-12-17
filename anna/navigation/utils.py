"""
Title: utils.py
Purpose:
Notes:
    * The navigator class oversees the environment, frame processing,
        and action selection
    * The navigator class handles transitioning from state to state
"""


# ============================================
#            get_new_navigator
# ============================================
def get_new_navigator():
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
    env = anna.utils.env.build_env()
    # Build the frame manager
    frameManager = anna.frames.utils.get_new_frame_manager()
    # Build the action manager
    actionManager = anna.actions.utils.get_new_action_manager()
    # Build the navigator
    if navParams.mode == "frameByFrame":
        navigator = FrameByFrameNavigator()
    return navigator
