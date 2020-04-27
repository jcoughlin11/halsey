"""
Title: base.py
Notes:
"""
import gin
from rich.progress import track

from halsey.io.logging import log
from halsey.io.write import save_checkpoint
from halsey.utils.setup import setup_checkpoint


# ============================================
#              BaseInstructor
# ============================================
@gin.configurable(whitelist=["trainParams"])
class BaseInstructor:
    """
    Doc string.
    """

    # -----
    # constructor
    # -----
    def __init__(self, navigator, memory, brain, trainParams):
        """
        Doc string.
        """
        self.navigator = navigator
        self.memory = memory
        self.brain = brain
        self.nEpisodes = trainParams["nEpisodes"]
        self.maxEpisodeSteps = trainParams["maxEpisodeSteps"]
        self.batchSize = trainParams["batchSize"]
        self.savePeriod = trainParams["savePeriod"]
        self.episode = 0
        self.episodeStep = 0
        self.checkpoint, self.checkpointManager = setup_checkpoint(self.brain)

    # -----
    # train
    # -----
    def train(self):
        """
        Doc string.
        """
        self.memory.pre_populate(self.navigator)
        episodes = track(range(self.nEpisodes), description="Training...")
        for self.episode in episodes:
            self.navigator.reset()
            for self.episodeStep in range(self.maxEpisodeSteps):
                experience = self.navigator.transition(self.brain, "train")
                self.memory.add(experience)
                sample = self.memory.sample(self.batchSize)
                self.brain.learn(sample)
                # Check for terminal state
                if experience[-1]:
                    break
            if (self.episode + 1) % self.savePeriod == 0:
                log(
                    "Saving episode: {}...".format(self.episode + 1),
                    silent=True,
                )
                save_checkpoint(self)

    # -----
    # get_instructor_state
    # -----
    def get_instructor_state(self):
        """
        Each new instructor child class should implement this if they
        introduce new stateful variables. This provides all of the
        stateful variables for saving.
        """
        statefulVars = {
            "episode": self.episode,
            "episodeStep": self.episodeStep,
        }
        return statefulVars
