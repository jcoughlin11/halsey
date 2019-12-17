"""
Title: utils.py
Purpose:
Notes:
    * The frame manager handles all image pre-processing and processing
"""


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
