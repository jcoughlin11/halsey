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
        self.startEpisode = 0
        self.startEpisodeStep = 0

    # -----
    # train
    # -----
    def train(self):
        """
        Doc string.
        """
        self.memory.pre_populate(self.game)
        episodes = track(
            range(self.startEpisode, self.nEpisodes), description="Training..."
        )
        for episode in episodes:
            self.game.reset()
            episodeSteps = range(self.startEpisodeStep, self.maxEpisodeSteps)
            for episodeStep in episodeSteps:
                experience = self.game.transition(self.brain, "train")
                self.memory.add(experience)
                sample = self.memory.sample(self.batchSize)
                self.brain.learn(sample)
                # Check for terminal state
                if experience[-1]:
                    break

    # -----
    # get_state
    # -----
    def get_state(self):
        """
        Gets the info from each object needed to continue training from
        where we left off. This is mostly internal counters and such.
        """
        internalState = {}
        internalState["gameState"] = self.game.get_state()
        internalState["memState"] = self.memory.get_state()
        internalState["brainState"] = self.brain.get_state()
        internalState["trainState"] = self._get_state()
        return internalState

    # -----
    # _get_state
    # -----
    def _get_state(self):
        """
        Saves trainer-specific variables that are not already contained
        in the parameter file or another object.
        """
        trainState = {}
        trainState["episode"] = self.startEpisode
        trainState["episodeStep"] = self.startEpisodeStep
        return trainState
