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
    # process
    # -----
    @abstractmethod
    def process(self, frame, newEpisode):
        """
        Driver routine for formatting the given game frame.
        """
        pass
