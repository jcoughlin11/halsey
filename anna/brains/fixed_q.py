"""
Title: fixed_q.py
Purpose: Contains the brain class for learning via the fixed-Q
            technique.
Notes:
"""


# ============================================
#                FixedQBrain
# ============================================
class FixedQBrain(QBrain):
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
    def __init__(self, networkParams, nActions, inputShape):
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
        super().__init__(networkParams, nActions, inputShape)
        # Build secondary network
        self.tNet = anna.networks.utils.build_network(
            self.arch,
            self.inputShape,
            self.nActions,
            self.optimizerName,
            self.lossName,
            self.learningRate,
        )
        # Make weights the same in both networks
        self.tNet.set_weights(self.qNet.get_weights())

    # -----
    # learn
    # -----
    def learn(self, memory):
        """
        In DQL the qTargets (labels) are determined from the same
        network that they are being used to update. As such, there can
        be a lot of noise due to values constantly jumping wildly. This
        affects the speed of convergence.

        In fixed-Q, a second network is used, called the target
        network. It's used to determine the qTargets (hence its name).

        These labels are then passed to the primary network so its
        weights can be updated. The weights in the primary network are
        copied over to the target network only every N steps. Due to
        this, there is far less jumping around in the target network,
        which makes its predicted labels more stable. This, in turn,
        speeds up convergence for the primary network.

        See Lillicrap et al. 2016.

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
        states, actions, rewards, nextStates, dones = memory.sample()
        # Use the target network to generate the qTargets
        qNext = self.tNet.predict_on_batch(nextStates)
        # Get the qTarget values according to dqn_learn
        qPred = self.qNet.predict_on_batch(states)
        qTarget = np.zeros(qPred.shape)
        doneInds = np.where(dones)
        nDoneInds = np.where(~dones)
        qTarget[doneInds, actions[doneInds]] = rewards[doneInds]
        qTarget[nDoneInds, actions[nDoneInds]] = rewards[
            nDoneInds
        ] + self.discountRate * np.amax(qNext[nDoneInds])
        qTarget[qTarget == 0] = qPred[qTarget == 0]
        # Get abs error
        absError = tf.reduce_sum(tf.abs(qTarget - qPred), axis=1)
        # Update the network weights
        loss = self.qNet.train_on_batch(
            states, qTarget, sample_weight=isWeights
        )
        return STUFF

    # -----
    # update
    # -----
    def update(self):
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
