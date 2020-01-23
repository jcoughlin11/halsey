"""
Title: vanilla.py
Purpose: Contains the original Deep-Q learning method from DeepMind.
Notes:
"""
import numpy as np
import tensorflow as tf

from .qbrain import QBrain


# ============================================
#               VanillaQBrain
# ============================================
class VanillaQBrain(QBrain):
    """
    Implements the original deep-q learning method from DeepMind.

    Attributes
    ----------
    See halsey.brains.dql.qbrain.QBrain

    Methods
    -------
    learn(memory, batchSize)
        Samples from the memory buffer and uses that sample to update
        the network weights.

    update()
        Updates the brain's state.
    """

    # -----
    # constructor
    # -----
    def __init__(self, brainParams, nActions, inputShape, channelsFirst):
        """
        Sets up the brain.

        Parameters
        ----------
        brainParams : halsey.utils.folio.Folio
            Contains the brain-related parameters read in from the
            parameter file.

        nActions : int
            The size of the game's action space. Determines the
            network's output shape.

        inputShape : list
            Contains the dimensions of the input to the network.

        channelsFirst : bool
            If True, then the first element of inputShape is the number
            of channels in the input. If False, then the last element of
            inputShape is assumed to be the number of channels.

        Raises
        ------
        None

        Returns
        -------
        None
        """
        # Call parent's constructor
        super().__init__(brainParams, nActions, inputShape, channelsFirst)

    # -----
    # learn
    # -----
    def learn(self, memory, batchSize):
        """
        Samples from the memory buffer and then uses that sample to
        update the network's weights.

        The network serves as an approximation to the full, very large
        Q-table for the game. Instead of updating entries to a Q-table,
        we update the weights in the network.

        In order to update the weights we need a **label** to compare
        to the network's **guess**. The estimates of the max discounted
        future rewards (qTarget) are the labels. These are obtained
        from the Bellmann equation.

        The Bellmann equation provides a way to estimate, via discounted
        future rewards obtained from the sample trajectories, how well
        we can do by playing optimally from the state that the chosen
        action brings us to. If this trajectory is bad, we lower the
        Q value for the current state-action pair. If it's good, then we
        increase it.

        We only change the entry for the current state-action pair
        because the current sample trajectory doesn't tell us anything
        about what would have happened had we chosen a different action
        for the current state, and we don't know the true Q-vectors
        ahead of time. To update those other entries, we need a
        different sample. This is why it takes so many training games to
        get a good Q-table (network).

        See algorithm 1 in [Minh13]_ for the calculation of qTarget.
        See `this page <https://keon.io/deep-q-learning/>`_ for
        implementation logic.

        Parameters
        ----------
        memory : halsey.memory.Memory
            Contains the buffer of experiences that the agent has had.
            The sample used to update the network weights is drawn from
            this buffer.

        batchSize : int
            The number of samples to draw from the memory buffer.

        Raises
        ------
        None

        Returns
        -------
        None
        """
        # Get sample of experiences
        states, actions, rewards, nextStates, dones = memory.sample(batchSize)
        # Get network's current guesses for Q values
        qPred = self.qNet.predict_on_batch(states)
        # Get qNext: estimate of best trajectory obtained by playing
        # optimally from the next state. This is used in the estimate
        # of Q-target (the labels)
        qNext = self.qNet.predict_on_batch(nextStates)
        # Update only the entry for the current state-action pair in the
        # vector of Q-values. For a terminal state it's just the reward,
        # and otherwise we use the Bellmann equation
        doneInds = np.where(dones)[0]
        nDoneInds = np.where(~dones)[0]
        # This third array is needed so I can get absError, otherwise
        # just the specified entries of qPred could be changed
        qTarget = np.zeros(qPred.shape)
        qTarget[doneInds, actions[doneInds].flatten()] = rewards[
            doneInds
        ].flatten()
        qTarget[nDoneInds, actions[nDoneInds].flatten()] = rewards[
            nDoneInds
        ].flatten() + self.discountRate * np.amax(
            tf.boolean_mask(qNext, ~(dones.flatten())), axis=1
        )
        # Fill in qTarget with the unaffected Q values. This is so the
        # TD error for those terms is 0, since they did not change.
        # Otherwise, the TD error for those terms would be equal to
        # the original Q value for that state-action entry
        qTarget[qTarget == 0] = qPred[qTarget == 0]
        # Get the absolute value of the TD error for use in PER. The
        # sum is so we only get 1 value per sample, since the priority
        # for each experience is just a float, not a sequence
        absError = tf.reduce_sum(tf.abs(qTarget - qPred), axis=1)
        # Update the network weights
        self.loss = self.qNet.train_on_batch(
            states, qTarget, sample_weight=memory.isWeights
        )

    # -----
    # update
    # -----
    def update(self):
        """
        Updates the brain's internal state (counters, non-primary
        networks, etc.). For vanilla Q-learning, there's nothing
        to update. This method is called by the trainer, which is
        agnostic to the learning method being used, which is why
        the method is needed here.

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
        pass
