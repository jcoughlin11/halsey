"""
Title: vanillafm.py
Purpose: Contains the VanillaFrameManager class.
Notes:
"""
import collections

import numpy as np

from halsey.utils.validation import register_option

from .baseprocessor import BaseFrameManager


# ============================================
#             VanillaFrameManager
# ============================================
@register_option
class VanillaFrameManager(BaseFrameManager):
    """
    This frame manager processes 2D RGB images, preparing them for
    input into the network.

    Attributes
    ----------
    frameStack : collections.deque
        Holds a trace of processed frames that is used for input into
        the neural network.

    Methods
    -------
    process_frame(frame, newEpisode=True)
        Driver function for handling a frame and adding it to the
        frame stack.

    See Also
    --------
    :py:class:`halsey.frames.baseprocessor.BaseFrameManager`
    """

    # -----
    # constructor
    # -----
    def __init__(self, frameParams):
        """
        Parameters
        ----------
        frameparams : halsey.utils.folio.Folio
            The frame-specific parameters as read-in from the parameter
            file.

        Raises
        ------
        None

        Returns
        -------
        None
        """
        super().__init__(frameParams)
        self.frameStack = None

    # -----
    # process_frame
    # -----
    def process_frame(self, frame, newEpisode=False):
        """
        Driver function for processing a frame and adding it to the
        stack of frames.

        Parameters
        ----------
        frame : np.ndarray
            The 2D image of the current game state.

        newEpisode : bool
            If True, we need to instantiate a new frame stack,
            otherwise we can add to the existing one.

        Raises
        ------
        None

        Returns
        -------
        stackedFrame : np.ndarray
            An array version of the deque frameStack.
        """
        # Preprocess the given state
        preprocessedFrame = self.preprocess_frame(frame)
        # Start fresh if this is a new episode
        if newEpisode:
            self.frameStack = collections.deque(
                [preprocessedFrame for i in range(self.traceLen)],
                maxlen=self.traceLen,
            )
        # Otherwise, add the frame to the stack
        else:
            self.frameStack.append(preprocessedFrame)
        # Create the tensorial version of the stack. Using axis=0 makes
        # an array with shape (traceLen, shrinkRows, shrinkCols) and
        # axis=2 gives (shrinkRows, shrinkCols, traceLen)
        if self.channelsFirst:
            stackedFrame = np.stack(self.frameStack, axis=0)
        else:
            stackedFrame = np.stack(self.frameStack, axis=2)
        return stackedFrame
