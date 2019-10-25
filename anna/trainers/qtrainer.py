"""
Title:   qtrainer.py
Purpose: Contains the QTrainer class.
Notes:
"""
import anna


# ============================================
#                 QTrainer
# ============================================
class QTrainer:
    """
    Handles choosing actions, interacting with the environment, and
    calling the learn methods.

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
    def __init__(self, trainParams):
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
        self.episode = 0
        self.startEpisode = 0
        self.earlyStop = False
        self.doneTraining = False
        self.episodeStep = 0

    # -----
    # train
    # -----
    def train(self, brain, memory, navigator):
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
            # Loop over the max number of steps allowed per episode
            for self.episodeStep in range(self.maxEpisodeSteps):
                # Check for early stopping
                self.earlyStop = anna.utils.endrun.check_early_stop()
                if self.earlyStop:
                    break
                # Transition to next state
                experience = navigator.transition(brain)
                # Save the experience
                memory.add(experience)
                # Update network weights
                brain.learn(memory, self.batchSize)
                # Update the brain's parameters (e.g., target q-network)
                brain.update()
                # Update memory, if applicable (e.g., priority weights)
                memory.update()
                # Update trainer's params
                self.update_params()
                # Check for terminal state
                if experience.done:
                    break
            # Break out of the training loop if needed
            if self.earlyStop:
                break
            # See if we need to save a checkpoint
            if self.episode % self.savePeriod == 0:
                yield brain, memory, navigator
        # If we get here, we're done
        self.doneTraining = True
        return brain, memory, navigator
