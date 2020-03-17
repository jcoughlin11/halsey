"""
Title: base.py
Notes:
"""
import queue

import gin
import numpy as np
import tensorflow as tf


# ============================================
#                BasePipelines
# ============================================
@gin.configurable(blacklist=["channelsFirst"])
class BasePipeline:
    """
    Doc string.

    Attributes
    ----------
    pass

    Methods
    -------
    pass
    """

    # -----
    # constructor
    # -----
    def __init__(self, channelsFirst, frameParams):
        """
        Parameters
        ----------
        pass

        Raises
        ------
        pass

        Returns
        -------
        pass
        """
        self.channelsFirst = channelsFirst
        self.normValue = frameParams["normValue"]
        self.traceLen = frameParams["traceLen"]
        self.offsetHeight = frameParams["offsetHeight"]
        self.offsetWidth = frameParams["offsetWidth"]
        self.cropHeight = frameParams["cropHeight"]
        self.cropWidth = frameParams["cropWidth"]
        self.frameStack = queue.deque(maxlen=self.traceLen)

    # -----
    # process
    # -----
    def process(self, frame, newEpisode):
        """
        Doc string.
        """
        # Normalize the frame. Shape should be (nRows, nCols, 3)
        frame = frame / self.normValue
        # Grayscale. Shape after grayscaling should be (nRows, nCols, 1)
        # frame is now a tensor after grayscaling
        frame = tf.image.rgb_to_grayscale(frame)
        # Cut out unncessary parts of the frame
        frame = tf.image.crop_to_bounding_box(
            frame,
            self.offsetHeight,
            self.offsetWidth,
            self.cropHeight,
            self.cropWidth,
        )
        # Remove the channels dimension to make stacking to the proper
        # shape easier. Gives a shape of (self.cropHeight, self.cropWidth)
        frame = tf.squeeze(frame)
        # If this is a new episode, there's no motion, so fill
        # deque with the frame. Otherwise, put the frame on top of the
        # deque. We save the numpy version to make indexing during
        # learning much easier
        if newEpisode:
            for _ in range(self.traceLen):
                self.frameStack.append(frame.numpy())
        else:
            self.frameStack.append(frame.numpy())
        # Return a tensorial version of the stack of frames to be used
        # by the network(s). Can't cast a deque to a tensor directly
        if self.channelsFirst:
            state = np.stack(self.frameStack, axis=0)
        else:
            state = np.stack(self.frameStack, axis=2)
        return state
