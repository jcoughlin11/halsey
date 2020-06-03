"""
Title: epsilongreedy.py
Notes:
"""
import gin
import numpy as np

from halsey.utils.endrun import endrun


# ============================================
#           EpsilonGreedyExplorer
# ============================================
@gin.configurable
class EpsilonGreedyExplorer:
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
        self.epsDecayRate = params["epsDecayRate"]
        self.epsilonStart = params["epsilonStart"]
        self.epsilonStop = params["epsilonStop"]
        self.decayStep = 0

    # -----
    # choose
    # -----
    def choose(self, brain, env, frameStack):
        """
        Doc string.
        """
        exploitProb = np.random.random()
        exploreProb = self.epsilonStop + (
            self.epsilonStart - self.epsilonStop
        ) * np.exp(-self.epsDecayRate * self.decayStep)
        self.decayStep += 1
        if exploreProb < 0.0:
            msg = f"Probability of exploring is negative: `{exploreProb}`"
            endrun(msg)
        elif exploreProb >= exploitProb:
            action = env.action_space.sample()
        else:
            action = np.argmax(brain.predict(frameStack).numpy())
        return action
