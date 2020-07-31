"""
Title: base.py
Notes:
"""
from abc import ABC
from abc import abstractmethod


# ============================================
#             BaseImagePipeline
# ============================================
class BaseImagePipeline(ABC):
    """
    The `imagePipeline` object handles processing the game frames into
    the form required by the neural networks.
    """

    # -----
    # constructor
    # -----
    def __init__(self, params):
        self.params = params

    # -----
    # normalize_frame
    # -----
    @abstractmethod
    def normalize_frame(self, frame):
        """
        Normalizes the pixel values in the frame to a given value.
        """
        pass

    # -----
    # grayscale
    # -----
    @abstractmethod
    def grayscale(self, frame):
        """
        Formats the image so that the number of color channels is one.
        """
        pass

    # -----
    # crop
    # -----
    @abstractmethod
    def crop(self, frame):
        """
        Cuts off unnecessary parts of the frame and then resizes the
        frame.
        """
        pass

    # -----
    # stack
    # -----
    @abstractmethod
    def stack(self, frame, newEpisode):
        """
        Puts multiple frames on top of one another to form a state.
        This is done to deal with the problem of time.
        """
        pass

    # -----
    # process
    # -----
    @abstractmethod
    def process(self, frame, newEpisode):
        """
        Driver routine for formatting the given game frame.
        """
        pass
