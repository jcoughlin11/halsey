"""
Title: base.py
Notes:
"""
import gin


# ============================================
#              BaseInstructor
# ============================================
@gin.configurable("baseTrainer", whitelist=["trainParams"])
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

    # -----
    # train
    # -----
    def train(self):
        """
        Doc string.
        """
        self.memory.pre_populate(self.navigator)
        for episode in range(self.nEpisodes):
            self.navigator.reset()
            for episodeStep in range(self.maxEpisodeSteps):
                experience = self.navigator.transition(self.brain, "train")
                self.memory.add(experience)
                sample = self.memory.sample(self.batchSize)
                self.brain.learn(sample)
                # Check for terminal state
                if experience[-1]:
                    break
