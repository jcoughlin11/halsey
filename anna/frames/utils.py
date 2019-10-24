"""
Title: utils.py
Purpose: Contains functions related to building a new frame manager.
Notes:
"""
from anna.frames.vanillaframemanager import VanillaFrameManager


# ============================================
#          get_new_frame_manager
# ============================================
def get_new_frame_manager(frameParams):
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
    if frameParams.mode == "vanilla":
        frameManager = VanillaFrameManager(frameParams)
    return frameManager
