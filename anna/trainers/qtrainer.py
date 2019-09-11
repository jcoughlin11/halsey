"""
Title:   qtrainer.py
Purpose: Contains the QTrainer class.
Notes:
"""


#============================================
#                 QTrainer
#============================================
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
    #-----
    # constructor
    #-----
    def __init__(self):
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
        self.done = False
        self.saveCheckpoint = False
        self.saveFinal = False
        self.params = self.initialize_params()
        self.frameHandler = FrameHandler()
        self.actionSelector = anna.exploration.utils.get_new_action_selector()

    #-----
    # train
    #-----
    def train(self, brain, memory):
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
        for self.params.episode in range(self.params.startEpisode, self.params.nEpisodes):
            # Reset to prepare for next episode after starting episode
            # since everything enters this loop as it should be
            if episode > self.params.startEpisode:
                self.reset_episode()
            # Loop over the max number of steps allowed per episode
            while self.params.episodeStep < self.params.maxEpSteps:
                # Check for early stopping
                if self.earlyStop = anna.utils.endrun.check_early_stop():
                    break
                # Transition to next state
                experience = self.transition()
                # Save the experience
                memory.save(experience)
                # Update network weights
                metrics = brain.learn()
                # Update the brain's parameters, if needed (e.g.,
                # update the target q-network)
                brain.update()
                # Update the trainer's params
                self.update_params()
                # Check for terminal state
                if experience.done:
                    # Get totals for episode metrics
                    self.update_episode_metrics()
                    # Go to next episode
                    break
                # Otherwise, set up for next step
                else:
                    state = nextState
            # See if we need to save a checkpoint
            if self.earlyStop or self.params.episode % self.savePeriod == 0:
                self.saveCheckpoint = True
                break
        # If we're early stopping we want to exit but not save the
        # final model because the episode loop didn't finish
        if self.earlyStop:
            self.done = True
        # The only way this triggers is if the episode loop finished
        # normally for the final episode, so we do want to save the
        # final model because it means training is done
        elif self.params.episode == self.params.nEpisodes - 1:
            self.done = True
            self.saveFinal = True
        return brain, memory
