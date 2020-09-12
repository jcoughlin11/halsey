"""
Title: base.py
Notes:
"""
from abc import ABC, abstractmethod
from queue import deque

from halsey.utils.register import register


# ============================================
#             BaseImagePipeline
# ============================================
class BaseImagePipeline(ABC):
    """
    The `imagePipeline` object handles processing the game frames into
    the form required by the neural networks.
    """

    # -----
    # subclass hook
    # -----
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register(cls)

    # -----
    # constructor
    # -----
    def __init__(self, params):
        self.params = params
        self.frameStack = deque(maxlen=self.params["traceLen"])
        self.dataFormat = None
        self.inputShape = None

    # -----
    # process
    # -----
    @abstractmethod
    def process(self, frame, newEpisode):
        """
        Driver routine for formatting the given game frame.
        """
        pass
