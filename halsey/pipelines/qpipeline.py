"""
Title: qpipeline.py
Notes:
"""
import numpy as np
import tensorflow as tf

from .base import BaseImagePipeline


# ============================================
#                  QPipeline
# ============================================
class QPipeline(BaseImagePipeline):
    """
    Pipeline used in Mnih et al. 2013.
    """

    # -----
    # normalize_frame
    # -----
    def normalize_frame(self, frame):
        """
        Normalizes the pixel values in the frame to a given value.
        """
        return frame / 255.0

    # -----
    # grayscale
    # -----
    def grayscale(self, frame):
        """
        Formats the image so that the number of color channels is one.
        """
        return tf.image.rgb_to_grayscale(frame)

    # -----
    # crop
    # -----
    def crop(self, frame):
        """
        Cuts off unnecessary parts of the frame and then resizes the
        frame.
        """
        frame = tf.image.crop_to_bounding_box(
            frame,
            self.params["offsetHeight"],
            self.params["offsetWidth"],
            self.params["cropHeight"],
            self.params["cropWidth"],
        )
        return frame

    # -----
    # stack
    # -----
    def stack(self, frame, newEpisode):
        """
        Puts multiple frames on top of one another to form a state.
        This is done to deal with the problem of time.
        """
        # Remove the channels dimension to make stacking to the proper
        # shape easier. Gives a shape of (self.cropHeight, self.cropWidth)
        frame = tf.squeeze(frame)
        # If this is a new episode, there's no motion, so fill
        # deque with the frame. Otherwise, put the frame on top of the
        # deque. We save the numpy version to make indexing during
        # learning much easier
        if newEpisode:
            for _ in range(self.params["traceLen"]):
                self.frameStack.append(frame.numpy())
        else:
            self.frameStack.append(frame.numpy())
        # Return a tensorial version of the stack of frames to be used
        # by the network(s). Can't cast a deque to a tensor directly
        if self.dataFormat == "channels_first":
            state = np.stack(self.frameStack, axis=0)
        else:
            state = np.stack(self.frameStack, axis=2)
        return state

    # -----
    # process
    # -----
    def process(self, frame, newEpisode):
        """
        Driver routine for formatting the given game frame.
        """
        # Normalize the frame. Shape should be (nRows, nCols, 3)
        frame = self.normalize_frame(frame)
        # Grayscale. Shape after grayscaling should be (nRows, nCols, 1)
        # frame is now a tensor after grayscaling
        frame = self.grayscale(frame)
        # Cut out unncessary parts of the frame
        frame = self.crop(frame)
        state = self.stack(frame, newEpisode)
        return state
