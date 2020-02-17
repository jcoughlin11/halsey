"""
Title: qtrainer.py
Purpose: Contains the QTrainer object.
Notes:
"""
import logging

from progress.spinner import Spinner

from halsey.utils.endrun import check_early_stop

from halsey.utils.validation import register_option

from .basetrainer import BaseTrainer


# ============================================
#                 QTrainer
# ============================================
@register_option
class QTrainer(BaseTrainer):
    """
    Contains the Deep Q-Learning training loop of [1]_.

    Attributes
    ----------
    trainGen : generator
        Contains the actual training loop. Having it as a generator
        allows for straightforward transfer of control back to the
        agent for tasks such as saving.

    Methods
    -------
    training_generator()
        A generator that contains the main training loop. It's a
        generator to allow for easy yielding back to the caller when
        it's time to save a checkpoint file.

    See Also
    --------
    :py:class:`~halsey.trainers.basetrainer.BaseTrainer`

    References
    ----------
    .. [1] Minh, V., **et al**., "Playing Atari with Deep
        Reinforcement Learning," CoRR, vol. 1312, 2013.
    """

    # -----
    # constructor
    # -----
    def __init__(self, trainParams, navigator, brain, memory, clArgs):
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
        super().__init__(trainParams, navigator, brain, memory, clArgs)
        self.trainGen = self.training_generator()

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
        spinner = Spinner()
        infoLogger = logging.getLogger("infoLogger")
        # Loop over the desired number of training episodes
        for self.episode in range(self.startEpisode, self.nEpisodes):
            msg = f"Episode: {self.episode+1} / {self.nEpisodes}"
            infoLogger.info(msg)
            # Loop over the max number of steps allowed per episode
            for self.episodeStep in range(self.maxEpisodeSteps):
                if not self.clArgs.silent:
                    spinner.next()
                # Check for early stopping
                self.earlyStop = check_early_stop()
                if self.earlyStop:
                    break
                # Transition to next state
                experience = self.navigator.transition(self.brain)
                # Save the experience
                self.memory.add(experience)
                # Get sample of experiences. This is done here and not
                # in the learn method because learn() is wrapped as a
                # tf.function. The tf docs recommend creating all
                # stateful variables outside the function and passing
                # them as args
                samples = self.memory.sample(self.batchSize)
                # Update network weights
                self.brain.learn(*samples)
                # Update brain's parameters (e.g., target q-network)
                self.brain.update()
                # Update memory (e.g., priority weights)
                self.memory.update()
                # Check for terminal state
                if experience[-1]:
                    break
            # Break out of the training loop if needed
            if self.earlyStop:
                break
            # See if we need to save a checkpoint
            if self.episode % self.savePeriod == 0:
                yield
