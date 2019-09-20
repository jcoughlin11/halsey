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
        pass

    #-----
    # train
    #-----
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
            # Reset to prepare for next episode after starting episode
            # since everything enters this loop as it should be
            if self.episode > self.startEpisode:
                self.reset_episode()
            # Loop over the max number of steps allowed per episode
            while self.episodeStep < self.maxEpSteps:
                # Check for early stopping
                if self.earlyStop = anna.utils.endrun.check_early_stop():
                    self.done = True
                    break
                # Transition to next state
                TRANSITION()
                # Save the experience
                memory.save()
                # Update network weights
                brain.learn(memory)
                # Update the brain's parameters (e.g., target q-network)
                brain.update()
                # Update memory, if applicable (e.g., priority weights)
                memory.update()
                # Update trainer's params
                self.update_params()
                # Check for terminal state
                if done:
                    # Get totals for episode metrics
                    self.update_episode_metrics()
                    # Go to next episode
                    break
                # Otherwise, set up for next step
                else:
                    state = nextState
            # See if we need to save a checkpoint. Don't save on early
            # stop since a checkpoint is always saved upon returning
            # below
            if self.episode % self.savePeriod == 0 and not self.earlyStop:
                yield STUFF 
        # If we get here, we're done
        self.done = True
        return STUFF 
