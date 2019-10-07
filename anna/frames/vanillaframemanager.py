"""
Title: vanillaframemanager.py
Purpose: Contains the standard frame manager.
Notes:
"""


#============================================
#           VanillaFrameManager
#============================================
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
    #-----
    # constructor
    #-----
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
        self.frameStack = None
        self.inputShape = [self.traceLen, self.shrinkRows, self.shrinkCols]

    #-----
    # process_frame
    #-----
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
                [preProcessedFrame for i in range(self.traceLen)], maxlen=self.traceLen
            )
        # Otherwise, add the frame to the stack
        else:
            self.frameStack.append(preprocessedFrame)
        # Create the tensorial version of the stack
        stackedFrame = np.stack(self.frameStack, axis=2)
        return stackedFrame

    #-----
    # preprocess_frame
    #-----
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
        greyFrame = skimage.color.rgb2grey(frame)
        # Crop the image b/c we don't need blank space or things on the
        # screen that aren't game objects
        croppedFrame = self.crop_frame(greyFrame)
        # Normalize the image
        normFrame = self.norm_frame(croppedFrame)
        # To reduce the computational complexity, we can shrink the image
        shrunkFrame = skimage.transform.resize(normFrame, [self.shrinkRows, self.shrinkCols])
        return shrunkFrame

    #-----
    # crop_frame
    #-----
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
            croppedFrame = frame[self.cropTop : -self.cropBot, self.cropLeft : -self.cropRight]
        elif self.cropBot == 0 and self.cropRight != 0:
            croppedFrame = frame[self.cropTop :, self.cropLeft : -self.cropRight]
        elif self.cropBot == 0 and self.cropRight == 0:
            croppedFrame = frame[self.cropTop :, self.cropLeft :]
        elif self.cropBot != 0 and self.cropRight == 0:
            croppedFrame = frame[self.cropTop : -self.cropBot, self.cropLeft :]
        else:
            raise ValueError
        return croppedFrame
