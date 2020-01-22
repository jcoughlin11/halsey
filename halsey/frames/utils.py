"""
Title: utils.py
Purpose: Handles the creation of a new frame manager object.
Notes:
    * The frame manager handles all image pre-processing and processing
"""
from .vanillafm import VanillaFrameManager


# ============================================
#          get_new_frame_manager
# ============================================
def get_new_frame_manager(frameParams):
    """
    Handles the creation of a new frame manager object of the desired
    type.

    Parameters
    ----------
    frameParams : halsey.utils.folio.Folio
        Contains the relevant frame manager parameters from the
        parameter file.

    Raises
    ------
    None

    Returns
    -------
    frameManager : halsey.frames.FrameManager
        The frame manager object.
    """
    if frameParams.mode == "vanilla":
        frameManager = VanillaFrameManager(frameParams)
    return frameManager
