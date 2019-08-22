"""
Title: frames.py
Author: Jared Coughlin
Date: 7/30/19
Purpose: Contains tools for handling and preprocessing images
Notes:
"""
import collections
import warnings

import numpy as np
import skimage


# Skimage produces a lot of warnings
warnings.filterwarnings("ignore")


# ============================================
#                 crop_frame
# ============================================
def crop_frame(frame, crop):
    """
    Handles the different cases for cropping the frame to the proper
    size. It doesn't matter whether crop[0] and/or crop[2] are zero or
    not because the first term in a slice is always included, whereas
    the last one is not.

    Parameters:
    -----------
        frame : ndarray
            The game frame

        crop : tuple
            The number of rows to chop off from the top and bottom and
            number of columns to chop off from the left and right.

    Raises:
    -------
        pass

    Returns:
    --------
        cropFrame : ndarray
            The cropped version of frame
    """
    cropFrame = None
    # Sanity check on crop sizes compared to image size
    if crop[0] >= frame.shape[0] or crop[1] >= frame.shape[0]:
        raise ValueError("Error, can't crop more rows than are in frame!")
    if crop[2] >= frame.shape[1] or crop[3] >= frame.shape[1]:
        raise ValueError("Error, can't crop more cols than are in frame!")
    if crop[0] + crop[1] >= frame.shape[0]:
        raise ValueError("Error, total crop from bot and top too big!")
    if crop[2] + crop[3] >= frame.shape[1]:
        raise ValueError("Error, total crop from left and right too big!")
    # Crop the frame
    if crop[1] != 0 and crop[3] != 0:
        cropFrame = frame[crop[0] : -crop[1], crop[2] : -crop[3]]
    elif crop[1] == 0 and crop[3] != 0:
        cropFrame = frame[crop[0] :, crop[2] : -crop[3]]
    elif crop[1] == 0 and crop[3] == 0:
        cropFrame = frame[crop[0] :, crop[2] :]
    elif crop[1] != 0 and crop[3] == 0:
        cropFrame = frame[crop[0] : -crop[1], crop[2] :]
    # Sanity check
    if cropFrame is None:
        raise ValueError("Error in crop_frame, cropFrame not set!")
    if sum(crop) != 0 and cropFrame.shape == frame.shape:
        raise ValueError(
            "Error in crop_frame, shapes equal when they shouldn't be!"
        )
    elif sum(crop) == 0 and cropFrame.shape != frame.shape:
        raise ValueError(
            "Error in crop_Frame, shapes not equal when they should be!"
        )
    return cropFrame


# ============================================
#             preprocess_frame
# ============================================
def preprocess_frame(frame, crop, shrink):
    """
    Handles grayscaling the frame and cropping the frame to the proper
    size.

    Parameters:
    -----------
        frame : ndarray
            The game image to crop and grayscale

        crop : tuple
            The number of rows to chop off both the top and bottom, the
            number of cols to chop off both the left and right. (top,
            bot, left, right).

        shrink : tuple
            The (x,y) size of the shrunk image.

    Raises:
    -------
        pass

    Returns:
    --------
        processed_frame : ndarray
            The grayscaled and cropped frame.
    """
    # Grayscale the image
    frame = skimage.color.rgb2grey(frame)
    # Crop the image b/c we don't need blank space or things on the
    # screen that aren't game objects
    frame = crop_frame(frame, crop)
    # Normalize the image
    frame = frame / 255.0
    # To reduce the computational complexity, we can shrink the image
    frame = skimage.transform.resize(frame, [shrink[0], shrink[1]])
    return frame


# ============================================
#               stack_frames
# ============================================
def stack_frames(frameStack, state, newEpisode, stackSize, crop, shrink, arch, traceLen):
    """
    Takes in the current state and preprocesses it. Then, it adds the
    processed frame to the stack of frames. Two versions of this stack
    are returned: a tensorial version (ndarray) and a deque for easy
    pushing and popping. The shape of the returned data is
    (nRows, nCols, stackSize). 

    Parameters:
    -----------
        frameStack : deque
            The deque version of the stack of processed frames.

        state : gym state
            This is effectively the raw frame from the game.

        newEpisode : bool
            If True, then we need to produce a clean deque and tensor.
            Otherwise, we can just add the given frame (state) to the
            stack.

        stackSize : int
            The number of frames to include in the stack.

        crop : tuple
            (top, bot, left, right) to chop off each edge of the frame.

        shrink : tuple
            (x,y) size of the shrunk frame.

    Raises:
    -------
        pass

    Returns:
    --------
        stacked_state : ndarray
            This is the tensorial version of the frame stack deque.

        frame_stack : deque
            The deque version of the stack of frames.
    """
    # Error check
    if not newEpisode and not isinstance(frameStack, collections.deque):
        print("Error, must pass existing stack if not starting a new episode!")
        sys.exit(1)
    # Preprocess the given state. Has shape = (shrinkRows, shrinkCols)
    state = preprocess_frame(state, crop, shrink)
    # Start fresh if this is a new episode
    if newEpisode:
        if arch == 'rnn1':
            frameStack = collections.deque([state for i in range(traceLen)], maxlen=traceLen)
        else:
            frameStack = collections.deque(
                [state for i in range(stackSize)], maxlen=stackSize
            )
    # Otherwise, add the frame to the stack
    else:
        frameStack.append(state)
    # Create the tensorial version of the stack. Has
    # shape = (shrinkRows, shrinkCols, stackSize) for non-RNN and
    # (traceLen, shrinkRows, shrinkCols) for RNN
    if arch == 'rnn1':
       stackedState = np.stack(frameStack, axis=0) 
    else:
        stackedState = np.stack(frameStack, axis=2)
    return stackedState, frameStack
