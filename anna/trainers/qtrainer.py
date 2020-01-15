"""
Title: qtrainer.py
Purpose: Contains the QTrainer object.
Notes:
"""
import anna


# ============================================
#                 QTrainer
# ============================================
class QTrainer:
    """
    Contains the Deep Q-Learning training loop of [1]_.

    Trainers are used to manage the training loop and keep track of
    the state of the training loop for potentially saving and resuming
    training at a later time.

    Attributes
    ----------
    batchSize : int
        The size of the sample to draw from the memory buffer. This
        sample is used during learning.

    brain : anna.brains.QBrain
        Contains the neural network(s) and learning method.

    earlyStop : bool
        If True, stop the training loop.

    episode : int
        The current episode number.

    episodeStep : int
        The current step within the current episode.

    maxEpisodeSteps : int
        The maximum number of steps allowed per episode.

    memory : int
        Holds the agent's experiences that are used in learning.

    navigator : anna.navigation.BaseNavigator
        Manages the game environment, processing game states, choosing
        actions, and performing actions in order to transition to the
        next state.

    nEpisodes : int
        The number of episodes to train for.

    savePeriod : int
        Save the agent's state every savePeriod episodes.

    startEpisode : int
        The number of the starting episode. Really only matters when
        resuming training from a previous run.

    trainGen : generator
        Contains the actual training loop. Having it as a generator
        allows for straightforward transfer of control back to the
        agent for tasks such as saving.

    Methods
    -------
    train()
        Abstracts away the calling of the training generator.

    pre_populate()
        Populates the memory buffer with experiences generated with
        randomly chosen actions. This avoids the empty memory problem
        at the start of training.

    References
    ----------
    .. [1] Minh, V., **et al**., "Playing Atari with Deep
        Reinforcement Learning," CoRR, vol. 1312, 2013.
    """

    # -----
    # constructor
    # -----
    def __init__(self, trainParams, navigator, brain, memory):
        """
        Creates an instance of the training generator.

        Parameters
        ----------
        trainParams : anna.utils.folio.Folio
            Contains training-specific data read in from the parameter
            file.

        navigator : anna.navigation.BaseNavigator
            Handles the game environment, processing game frames,
            choosing actions, and transitioning from one state to the
            next.

        brain : anna.brains.QBrain
            Contains the neural network(s) and the learning method.

        memory : anna.memory.Memory
            Contains the buffer of experiences used during learning.

        Raises
        ------
        None

        Returns
        -------
        None
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
        self.episodeStep = 0
        self.trainGen = self.training_generator()

    # -----
    # train
    # -----
    def train(self):
        """
        Abstracts away the use of the training generator.

        Parameters
        ----------
        None

        Raises
        ------
        None

        Returns
        -------
        trainGen : generator
            A generator that executes the main training loop.
        """
        return self.trainGen

    # -----
    # training_generator
    # -----
    def training_generator(self):
        """
        The main deep Q-learning training loop.

        Parameters
        ----------
        None

        Raises
        ------
        None

        Returns
        -------
        None
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
                # Update brain's parameters (e.g., target q-network)
                self.brain.update()
                # Update memory (e.g., priority weights)
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

    # -----
    # pre_populate
    # -----
    def pre_populate(self):
        """
        Uses randomly chosen actions to generate experiences to fill
        the memory buffer. This is to avoid the empty memory problem.

        Parameters
        ----------
        None

        Raises
        ------
        None

        Returns
        -------
        None
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
