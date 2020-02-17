"""
Title: vanilla.py
Purpose: Contains the original Deep-Q learning method from DeepMind.
Notes:
"""
import numpy as np
import tensorflow as tf

from halsey.utils.validation import register_option

from ..basebrain import BaseBrain


# ============================================
#               VanillaQBrain
# ============================================
@register_option
class VanillaQBrain(BaseBrain):
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
    """

    # -----
    # constructor
    # -----
    def __init__(self, brainParams):
        """
        Sets up the brain.

        Parameters
        ----------
        brainParams : halsey.utils.folio.Folio
            Contains the brain-related parameters read in from the
            parameter file.

        Raises
        ------
        None

        Returns
        -------
        None
        """
        # Call parent's constructor
        super().__init__(brainParams)

    # -----
    # learn
    # -----
    def learn(self):
        """
        Uses the sample to update the network's weights via the
        Q-learning method.

        The network serves as an approximation to the full Q-table for
        the game. Instead of updating entries to a Q-table, we update
        the weights in the network.

        In order to update the weights we need a **"label"** to compare
        to the network's **guess**. The estimates of the max discounted
        future rewards (qTarget) are the labels. These are obtained
        from the Bellmann equation.

        The Bellmann equation provides a way to estimate, via discounted
        future rewards, how well we can do by playing optimally from the
        state that the chosen action brings us to. If this trajectory is
        bad, we lower the Q value for the current state-action pair. If
        it's good, then we increase it.

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
        pass

        Raises
        ------
        None

        Returns
        -------
        None
        """
        # Get network's current guesses for Q values
        qGuess = self.qNet(states, training=True)
        # Get the "labels": the estimates of how well we can do by
        # playing optimally from the state our chosen action brought
        # us to
        qNext = self.qNet(nextStates)
