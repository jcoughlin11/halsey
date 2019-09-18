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
    def __init__(self, env, trainParams, frameParams, exploreParams):
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
        self.env            = env
        self.episode        = 0
        self.earlyStop      = False
        self.done           = False
        self.saveCheckpoint = False
        self.startEpisode   = trainParams.startEpisode
        self.state          = self.env.reset()
        self.episodeStep    = trainParams.episodeStep
        self.nEpisodes      = trainParams.nEpisodes
        self.maxEpSteps     = trainParams.maxEpSteps
        self.frameHandler   = FrameHandler(frameParams)
        self.actionSelector = anna.exploration.utils.get_new_action_selector(exploreParams)

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
                experience = self.transition(brain)
                # Save the experience
                memory.save(self.state, experience)
                # Update network weights
                metrics = brain.learn(memory)
                # Update the brain's parameters (e.g., target q-network)
                brain.update()
                # Update trainer's params
                self.update_params()
                # Check for terminal state
                if experience.done:
                    # Get totals for episode metrics
                    self.update_episode_metrics()
                    # Go to next episode
                    break
                # Otherwise, set up for next step
                else:
                    self.state = experience['nextState']
            # See if we need to save a checkpoint. Don't save on early
            # stop since a checkpoint is always saved upon returning
            # below
            if self.episode % self.savePeriod == 0 and not self.earlyStop:
                self.saveCheckpoint = True
                yield brain, memory
        # If we get here, we're done
        self.done = True
        self.saveCheckpoint = True
        return brain, memory

    #-----
    # reset_episode
    #-----
    def reset_episode(self):
        """
        Resets everything (including the environment) for a new
        episode.

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
        self.state = self.env.reset()
        self.episodeStep = 0
        self.saveCheckpoint = False

    #-----
    # update_params
    #-----
    def update_params(self):
        """
        Handles incrementing the counters.

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
        self.episodeStep += 1

    #-----
    # update_episode_metrics
    #-----
    def update_episode_metrics(self):
        """
        Handles things like getting the total episode rewards.

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
    # transition
    #-----
    def transition(self, brain):
        """
        Uses the chosen strategy to select an action, take the action,
        and receive experience from the environment.

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
        # Choose an action using the desired action-selection scheme
        action = self.actionSelector.choose_action(self.state, brain)
        # Take the action
        nextState, reward, done, _ = self.env.step(action)
        # Package all of this into an experience and return
        experience = {}
        experience['action'] = action
        experience['reward'] = reward
        experience['nextState'] = nextState
        experience['done'] = done
        return experience
