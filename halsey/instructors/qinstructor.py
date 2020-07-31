"""
Title: qinstructor.py
Notes:
"""
from .base import BaseInstructor


# ============================================
#                QInstructor
# ============================================
class QInstructor(BaseInstructor):
    """
    Contains the training loop from Mnih et al. 2013.
    """

    # -----
    # train
    # -----
    def train(self):
        """
        For each episode, the agent chooses an action, takes the
        action, transitions to the next game state, stores the
        resulting experience in the memory buffer, samples randomly
        from the memory buffer, and then uses that sample to update
        the network weights.
        """
        self.brain.pre_populate(self.navigator)
        for episode in range(self.params["nEpisodes"]):
            self.navigator.reset()
            for episodeStep in range(self.params["maxEpisodeSteps"]):
                experience = self.navigator.transition(self.brain, "train")
                self.brain.add_memory(experience)
                self.brain.learn()
                # Check for terminal state
                if experience[-1]:
                    break
