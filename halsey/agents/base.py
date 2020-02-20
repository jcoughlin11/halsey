"""
Title:      base.py
Purpose:    Contains the base agent class.
Notes:
"""


# ============================================
#                 BaseAgent
# ============================================
class BaseAgent:
    """
    Contains the training loop, several convenience methods, and those
    attributes that are common to all agents.

    Parameters
    ----------
    pass

    Methods
    -------
    pass
    """

    # -----
    # constructor
    # -----
    def __init__(self, model, memory, navigator, trainParams):
        """
        Parameters
        ----------
        pass

        Raises
        ------
        pass

        Returns
        -------
        pass
        """
        self.model = model
        self.memory = memory
        self.navigator = navigator
        self.nEpisodes = trainParams["nEpisodes"]
        self.maxEpisodeSteps = trainParams["maxEpisodeSteps"]
        self.episode = 0
        self.episodeStep = 0

    # -----
    # train
    # -----
    def train(self):
        """
        Contains the Deep Q-Learning training loop of [1]_.

        Parameters
        ----------
        pass

        Raises
        ------
        pass

        Returns
        -------
        pass

        References
        ----------
        .. [1] Minh, V., **et al**., "Playing Atari with Deep
            Reinforcement Learning," CoRR, vol. 1312, 2013.
        """
        for self.episode in range(self.startEpisode, self.nEpisodes):
            for self.episodeStep in range(self.maxEpisodeSteps):
                experience = self.navigator.transition(self.model)
                self.memory.add(experience)
                trainData = self.memory.sample()
                self.model.learn(trainData)
                self.memory.update()
                self.model.update()
                # Check for terminal state
                if experience[-1]:
                    self.navigator.reset()
                    break
