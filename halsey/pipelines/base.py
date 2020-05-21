"""
Title: base.py
Notes:
"""
import queue

import numpy as np
import tensorflow as tf


# ============================================
#                BasePipelines
# ============================================
class BasePipeline:
    """
    Doc string.
    """

    # -----
    # constructor
    # -----
    def __init__(self, channelsFirst, params):
        """
        Doc string.
        """
        self.channelsFirst = channelsFirst
        self.normValue = params["normValue"]
        self.traceLen = params["traceLen"]
        self.offsetHeight = params["offsetHeight"]
        self.offsetWidth = params["offsetWidth"]
        self.cropHeight = params["cropHeight"]
        self.cropWidth = params["cropWidth"]
        self.frameStack = queue.deque(maxlen=self.traceLen)

    # -----
    # inputShape
    # -----
    @property
    def inputShape(self):
        """
        Doc string.
        """
        if self.channelsFirst:
            shape = [self.traceLen, self.cropHeight, self.cropWidth]
        else:
            shape = [self.cropHeight, self.cropWidth, self.traceLen]
        return shape

    # -----
    # normalize_frame
    # -----
    def normalize_frame(self, frame):
        """
        Doc string.
        """
        return frame / self.normValue

    # -----
    # grayscale
    # -----
    def grayscale(self, frame):
        """
        Doc string.
        """
        return tf.image.rgb_to_grayscale(frame)

    # -----
    # crop
    # -----
    def crop(self, frame):
        """
        Doc string.
        """
        frame = tf.image.crop_to_bounding_box(
            frame,
            self.offsetHeight,
            self.offsetWidth,
            self.cropHeight,
            self.cropWidth,
        )
        return frame

    # -----
    # stack
    # -----
    def stack(self, frame, newEpisode):
        """
        Doc string.
        """
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

    # -----
    # process
    # -----
    def process(self, frame, newEpisode):
        """
        Doc string.
        """
        # Normalize frame. Shape should be (nRows, nCols, nChannels)
        frame = self.normalize_frame(frame)
        # Grayscale. Shape after grayscaling should be (nRows, nCols, 1)
        # frame is now a tensor after grayscaling
        frame = self.grayscale(frame)
        # Cut out unncessary parts of the frame
        frame = self.crop(frame)
        state = self.stack(frame, newEpisode)
        return state
