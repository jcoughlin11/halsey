"""
Title: vanillafm.py
Purpose:
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
    Doc string.

    Attributes:
    -----------
        pass

    Methods:
    --------
        pass
    """

    # -----
    # constructor
    # -----
    def __init__(self, frameParams):
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
        frame = frame / 255.0
        return frame
