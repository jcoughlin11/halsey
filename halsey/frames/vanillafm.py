"""
Title: vanillafm.py
Purpose: Contains the VanillaFrameManager class.
Notes:
"""
import collections

import numpy as np
from skimage import color
from skimage import transform


# ============================================
#           VanillaFrameManager
# ============================================
class VanillaFrameManager:
    """
    This frame manager processes 2D RGB images, preparing them for
    input into the network.

    Attributes
    ----------
    channelsFirst : bool
        If True, the number of channels (in this case, the trace
        length) is the first element of the input shape. Otherwise,
        the trace length will be the last element.

    cropBot : int
        The number of rows to cut off from the bottom of the image.

    cropLeft : int
        The number of columns to cut off from the left of the image.

    cropRight : int
        The number of columns to cut off from the right of the image.

    cropTop : int
        The number of rows to cut off from the top of the image.

    frameStack : collections.deque
        Holds a trace of processed frames that is used for input into
        the neural network.

    inputShape : list
        The dimensions of the input into the neural network. Is either
        NCHW or NHWC, where N is the batch size, C is the number of
        channels (the trace length), H is the height (number of rows),
        and W is the width (number of columns).

    shrinkCols : int
        The number of columns to use in the shrunk-down image.

    shrinkRows : int
        The number of rows to use in the shrunk-down image.

    traceLen : int
        The number of frames to stack together to form one input.

    Methods
    -------
    process_frame(frame, newEpisode=True)
        Driver function for handling a frame and adding it to the
        frame stack.

    preprocess_frame(frame)
        Driver function for greyscaling, re-sizing, and normalizing the
        frame.

    crop_frame(frame)
        Handles re-sizing the frame.

    norm_frame(frame)
        Normalizes the pixel values in the frame.
    """

    # -----
    # constructor
    # -----
    def __init__(self, frameParams):
        """
        Sets the state and creates the input shape.

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
        self.cropBot = frameParams.cropBot
        self.cropLeft = frameParams.cropLeft
        self.cropRight = frameParams.cropRight
        self.cropTop = frameParams.cropTop
        self.shrinkCols = frameParams.shrinkCols
        self.shrinkRows = frameParams.shrinkRows
        self.traceLen = frameParams.traceLen
        self.channelsFirst = frameParams.channelsFirst
        self.frameStack = None
        if self.channelsFirst:
            self.inputShape = [self.traceLen, self.shrinkRows, self.shrinkCols]
        else:
            self.inputShape = [self.shrinkRows, self.shrinkCols, self.traceLen]

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

    # -----
    # preprocess_frame
    # -----
    def preprocess_frame(self, frame):
        """
        Driver function for greyscaling, re-sizing, and normalizing the
        frame.

        Parameters
        ----------
        frame : np.ndarray

        Raises
        ------
        None

        Returns
        -------
        shrunkFrame : np.ndarray
            The fully processed frame. It's been greyscaled, re-sized,
            and normalized.
        """
        # Grayscale the image
        greyFrame = color.rgb2grey(frame)
        # Crop the image b/c we don't need blank space or things on the
        # screen that aren't game objects
        croppedFrame = self.crop_frame(greyFrame)
        # Normalize the image
        normFrame = self.norm_frame(croppedFrame)
        # To reduce the computational complexity, we can shrink the image
        shrunkFrame = transform.resize(
            normFrame, [self.shrinkRows, self.shrinkCols]
        )
        return shrunkFrame

    # -----
    # crop_frame
    # -----
    def crop_frame(self, frame):
        """
        Handles re-sizing the frame.

        Parameters
        ----------
        frame : np.ndarray
            The image of the current game state.

        Raises
        ------
        None

        Returns
        -------
        croppedFrame : np.ndarray
            The re-sized frame.
        """
        # Crop the frame
        if self.cropBot != 0 and self.cropRight != 0:
            croppedFrame = frame[
                self.cropTop : -self.cropBot, self.cropLeft : -self.cropRight
            ]
        elif self.cropBot == 0 and self.cropRight != 0:
            croppedFrame = frame[
                self.cropTop :, self.cropLeft : -self.cropRight
            ]
        elif self.cropBot == 0 and self.cropRight == 0:
            croppedFrame = frame[self.cropTop :, self.cropLeft :]
        elif self.cropBot != 0 and self.cropRight == 0:
            croppedFrame = frame[self.cropTop : -self.cropBot, self.cropLeft :]
        else:
            raise ValueError
        return croppedFrame

    # -----
    # norm_frame
    # -----
    def norm_frame(self, frame):
        """
        Handles normalizing the frame.

        Parameters
        ----------
        frame : np.ndarray
            The current game image.

        Raises
        ------
        None

        Returns
        -------
        frame : np.ndarray
            The normalized frame.
        """
        frame = frame / 255.0
        return frame
