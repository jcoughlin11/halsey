"""
Title: base.py
Notes:
"""
import gin
from rich.progress import track


# ============================================
#                 BaseTrainer
# ============================================
@gin.configurable
class BaseTrainer:
    """
    Doc string.
    """

    # -----
    # constructor
    # -----
    def __init__(self, game, memory, brain, chkpt, chkptMgr, params):
        """
        Doc string.
        """
        self.game = game
        self.memory = memory
        self.brain = brain
        self.nEpisodes = params["nEpisodes"]
        self.maxEpisodeSteps = params["maxEpisodeSteps"]
        self.batchSize = params["batchSize"]
        self.checkpoint = chkpt
        self.checkpointManager = chkptMgr

    # -----
    # train
    # -----
    def train(self):
        """
        Doc string.
        """
        self.memory.pre_populate(self.game)
        for episode in track(range(self.nEpisodes), description="Training..."):
            self.game.reset()
            for episodeStep in range(self.maxEpisodeSteps):
                experience = self.game.transition(self.brain, "train")
                self.memory.add(experience)
                sample = self.memory.sample(self.batchSize)
                self.brain.learn(sample)
                # Check for terminal state
                if experience[-1]:
                    break
