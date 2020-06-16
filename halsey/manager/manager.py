"""
Title: manager.py
Notes:
"""
import gin

from halsey.io.write import save_model
from halsey.utils.misc import io_check
from halsey.utils.misc import sanitize_path
from halsey.utils.setup import get_trainer


# ============================================
#                   Manager
# ============================================
@gin.configurable("general")
class Manager:
    """
    Doc string.
    """

    # -----
    # constructor
    # -----
    def __init__(self, clArgs, params):
        """
        Doc string.
        """
        self.outputDir = sanitize_path(params["outputDir"])
        self.doTraining = params["train"]
        self.doTesting = params["test"]
        self.doAnalysis = params["analyze"]
        self.clArgs = clArgs

    # -----
    # train
    # -----
    def train(self):
        """
        Doc string.
        """
        trainer = get_trainer()
        trainer.train()
        save_model(self, trainer)

    # -----
    # pre_flight_check
    # -----
    def pre_flight_check(self):
        """
        Doc string.
        """
        io_check(self.outputDir)
