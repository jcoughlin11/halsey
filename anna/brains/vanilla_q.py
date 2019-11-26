"""
Title: vanillaq.py
Purpose: Contains the VanillaQBrain class.
Notes:
"""
import numpy as np
import tensorflow as tf

from anna.brains.qbrain import QBrain


# ============================================
#               VanillaQBrain
# ============================================
class VanillaQBrain(QBrain):
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
    def __init__(self, networkParams, nActions, frameManager):
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
        # Call parent's constructor
        super().__init__(networkParams, nActions, frameManager)

    # -----
    # learn
    # -----
    def learn(self, memory, batchSize):
        """
        The estimates of the max discounted future rewards (qTarget) are
        the "labels" assigned to the input states.

        Basically, the network holds the current beliefs for how well
        we can do by taking a certain action in a certain state. The
        Bellmann equation provides a way to estimate, via discounted
        future rewards obtained from the sample trajectories, how well
        we can do playing optimally from the state that the chosen
        action brings us to. If this trajectory is bad, we lower the
        Q value for the current state-action pair. If it's good, then we
        increase it.

        But we only change the entry for the current state-action pair
        because the current sample trajectory doesn't tell us anything
        about what would have happened had we chosen a different action
        for the current state, and we don't know the true Q-vectors
        ahead of time. To update those other entries, we need a
        different sample. This is why it takes so many training games to
        get a good Q-table (network).

        See Mnih13 algorithm 1 for the calculation of qTarget.
        See https://keon.io/deep-q-learning/ for implementation logic.

        Note: The way OpenAI baselines does this is better.

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
        # Get sample of experiences
        states, actions, rewards, nextStates, dones = memory.sample(batchSize)
        # Get network's current guesses for Q values
        qPred = self.qNet.predict_on_batch(states)
        # Get qNext: estimate of best trajectory obtained by playing
        # optimally from the next state. This is used in the estimate
        # of Q-target
        qNext = self.qNet.predict_on_batch(nextStates)
        # Update only the entry for the current state-action pair in the
        # vector of Q-values that corresponds to the chosen action. For
        # a terminal state it's just the reward, and otherwise we use
        # the Bellmann equation
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
        # Get the absolute value of the TD error for use in per. The
        # sum is so we only get 1 value per sample, since the priority
        # for each experience is just a float, not a sequence
        absError = tf.reduce_sum(tf.abs(qTarget - qPred), axis=1)
        # Update the network weights
        loss = self.qNet.train_on_batch(
            states, qTarget, sample_weight=isWeights
        )
        assert False
        # return STUFF

        # -----
        # update
        # -----
        def update(self):
            """
            Updates the brain's internal state (counters, non-primary
            networks, etc.). For vanilla Q-learning, there's nothing
            to update.

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
