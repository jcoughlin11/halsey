"""
Title: qtrainer.py
Purpose:
Notes:
"""
import anna


# ============================================
#                 QTrainer
# ============================================
class QTrainer:
    """
    Doc string.

    Attributes:
    -----------
        pass

    Methods:
    --------
        pass
    """

    # -----
    # constructor
    # -----
    def __init__(self, trainParams, navigator, brain, memory):
        """
        Doc string.

        Parameters:
        -----------
            pass

        Raises:
        -------
            pass

        Returns:
        --------
            pass
        """
        self.nEpisodes = trainParams.nEpisodes
        self.maxEpisodeSteps = trainParams.maxEpisodeSteps
        self.batchSize = trainParams.batchSize
        self.savePeriod = trainParams.savePeriod
        self.navigator = navigator
        self.brain = brain
        self.memory = memory
        self.episode = 0
        self.startEpisode = 0
        self.earlyStop = False
        self.doneTraining = False
        self.episodeStep = 0

    # -----
    # train
    # -----
    def train(self):
        """
        Doc string.

        Parameters:
        -----------
            pass

        Raises:
        -------
            pass

        Returns:
        --------
            pass
        """
        # Loop over the desired number of training episodes
        for self.episode in range(self.startEpisode, self.nEpisodes):
            print(
                "Episode: {} / {}".format(self.episode, self.nEpisodes),
                end="\r",
            )
            # Loop over the max number of steps allowed per episode
            for self.episodeStep in range(self.maxEpisodeSteps):
                print(
                    "Step: {} / {}".format(
                        self.episodeStep, self.maxEpisodeSteps
                    ),
                    end="\r",
                )
                # Check for early stopping
                self.earlyStop = anna.utils.endrun.check_early_stop()
                if self.earlyStop:
                    break
                # Transition to next state
                experience = self.navigator.transition(self.brain)
                # Save the experience
                self.memory.add(experience)
                # Update network weights
                self.brain.learn(self.memory, self.batchSize)
                # Update the brain's parameters (e.g., target q-network)
                self.brain.update()
                # Update memory, if applicable (e.g., priority weights)
                self.memory.update()
                # Check for terminal state
                if experience.done:
                    break
            # Break out of the training loop if needed
            if self.earlyStop:
                break
            # See if we need to save a checkpoint
            if self.episode % self.savePeriod == 0:
                yield
        # If we get here, we're done
        self.doneTraining = True

    # -----
    # pre_populate
    # -----
    def pre_populate(self):
        """
        Doc string.

        Parameters:
        -----------
            pass

        Raises:
        -------
            pass

        Returns:
        --------
            pass
        """
        # Reset the environment
        self.navigator.reset()
        # Loop over the desired number of sample experiences
        for i in range(self.memory.pretrainLen):
            experience = self.navigator.transition(mode="random")
            # Add experience to memory
            self.memory.add(experience)
        # Reset the navigator
        self.navigator.reset()
