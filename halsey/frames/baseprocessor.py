"""
Title: baseprocessor.py
Purpose: Contains the BaseFrameManager class.
Notes:
"""
import logging
import sys

from skimage import color
from skimage import transform


# ============================================
#              BaseFrameManager
# ============================================
class BaseFrameManager:
    """
    Acts as a container for all of the common attributes and methods
    for the various frame manager objects.

    A frame manager is, at its heart, the image processing pipeline
    that converts the images from the game into a format useable by the
    agent for learning.

    Attributes
    ----------
    cropBot : int
        The number of rows to cut off from the bottom of the image.

    cropLeft : int
        The number of columns to cut off from the left of the image.

    cropRight : int
        The number of columns to cut off from the right of the image.

    cropTop : int
        The number of rows to cut off from the top of the image.

    shrinkCols : int
        The number of columns to use in the shrunk-down image.

    shrinkRows : int
        The number of rows to use in the shrunk-down image.

    traceLen : int
        The number of frames to stack together to form one input.

    channelsFirst : bool
        Determines input shape. If True, the shape has the form NCHW,
        otherwise the shape is NHWC. This is used when stacking the
        frames together in order to know which dimension to stack
        along.

    Methods
    -------
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
        try:
            assert self.cropBot >= 0
            assert self.cropLeft >= 0
            assert self.cropRight >= 0
            assert self.cropTop >= 0
            assert self.shrinkCols >= 0
            assert self.shrinkRows >= 0
            assert self.traceLen >= 0
        except AssertionError:
            infoLogger = logging.getLogger("infoLogger")
            errorLogger = logging.getLogger("errorLogger")
            infoLogger.info("Error: Negative frame resizing parameter(s).")
            errorLogger.exception("Negative frame resizing parameter(s).")
            sys.exit(1)

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
