"""
Title: basetrainer.py
Purpose: Contains the BaseTrainer class.
Notes:
"""


# ============================================
#                BaseTrainer
# ============================================
class BaseTrainer:
    """
    Holds the attributes common to all training loops as well as
    several convenience functions utilized by all trainers.

    Trainers are used to manage the training loop and keep track of
    the state of the training loop for potentially saving and resuming
    training at a later time.

    Attributes
    ----------
    batchSize : int
        The size of the sample to draw from the memory buffer. This
        sample is used during learning.

    brain : halsey.brains.QBrain
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

    navigator : halsey.navigation.BaseNavigator
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

    Methods
    -------
    train()
        Abstracts away the calling of the training generator.

    pre_populate()
        Populates the memory buffer with experiences generated with
        randomly chosen actions. This avoids the empty memory problem
        at the start of training.
    """

    # -----
    # constructor
    # -----
    def __init__(self, trainParams, navigator, brain, memory):
        """
        Creates an instance of the training generator.

        Parameters
        ----------
        trainParams : halsey.utils.folio.Folio
            Contains training-specific data read in from the parameter
            file.

        navigator : halsey.navigation.BaseNavigator
            Handles the game environment, processing game frames,
            choosing actions, and transitioning from one state to the
            next.

        brain : halsey.brains.QBrain
            Contains the neural network(s) and the learning method.

        memory : halsey.memory.Memory
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
