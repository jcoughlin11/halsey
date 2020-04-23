"""
Title: base.py
Notes:
"""
import gin
from rich.progress import track
import tensorflow as tf

from halsey.io.logging import log
from halsey.io.write import save_checkpoint


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
        self.checkpoint, self.checkpointManager = self.setup_checkpoint()

    # -----
    # setup_checkpoint
    # -----
    def setup_checkpoint(self):
        """
        Doc string.

        See: https://www.tensorflow.org/guide/checkpoint
        """
        # Add optimizer
        checkpoint = tf.train.Checkpoint(optimizer=self.brain.optimizer)
        # Add network(s). This is how attributes are added to the
        # checkpoint object in the tf source
        for i, net in enumerate(self.brain.nets):
            checkpoint.__setattr__("net" + str(i), net)
        manager = tf.train.CheckpointManager(checkpoint, ".", max_to_keep=3)
        return checkpoint, manager

    # -----
    # train
    # -----
    def train(self):
        """
        Doc string.
        """
        self.memory.pre_populate(self.navigator)
        for episode in track(range(self.nEpisodes), description="Training..."):
            self.navigator.reset()
            for episodeStep in range(self.maxEpisodeSteps):
                experience = self.navigator.transition(self.brain, "train")
                self.memory.add(experience)
                sample = self.memory.sample(self.batchSize)
                self.brain.learn(sample)
                # Check for terminal state
                if experience[-1]:
                    break
            if (episode + 1) % self.savePeriod == 0:
                log("Saving episode: {}...".format(episode + 1))
                save_checkpoint(self)
