"""
Title: manager.py
Notes:
"""
import gin

from halsey.utils.misc import sanitize_path
from halsey.utils.setup import get_analyst
from halsey.utils.setup import get_tester
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
    def __init__(self, params):
        """
        Doc string.
        """
        self.outputDir  = sanitize_path(params["outputDir"])
        self.doTraining = params["train"]
        self.doTesting  = params["test"]
        self.doAnalysis = params["analyze"]

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
    # test
    # -----
    def test(self):
        """
        Doc string.
        """
        tester = get_tester()
        tester.test()

    # -----
    # analyze
    # -----
    def analyze(self):
        """
        Doc string.
        """
        analyst = get_analyst()
        analyst.analyze()
