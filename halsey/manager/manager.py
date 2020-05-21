"""
Title: manager.py
Notes:
"""
import gin

from halsey.utils.misc import create_output_directory
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

    # -----
    # io_check
    # -----
    def io_check(self):
        """
        Makes sure output dir can be made if starting from scratch
        or that output dir exists and has all the right files in it
        if continuing training or testing or analyzing. That kind of
        stuff. This prevents lots of time being spent on something only
        to find out that none of the work can be saved for some reason.
        Better to find that out before doing all the work.
        """
        create_output_directory(self.outputDir)
