"""
Title: double_dqn.py
Purpose: Contains the brain class for learning via the double-DQN
            technique.
Notes:
"""


#============================================
#              DoubleDqnBrain
#============================================
class DoubleDqnBrain(FixedQBrain):
    """
    Doc string.

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

    # -----
    # learn
    # -----
    def learn(self, memory):
        """
        Double dqn attempts to deal with the following issue: when we
        choose the action that gives rise to the highest Q value for the
        next state, how do we know that that's actually the best action?

        Since we're learning from experience, our estimated Q values
        depend on which actions have been tried and which neighboring
        states have been visited.

        As such, double dqn separates out the estimate of the Q value
        and the determination of the best action to take at the next
        state.
        
        We use our primary network to choose an action for the
        next state and then pass that action to our target network,
        which handles calculating the target Q value.

        For non-terminal states, the target value is:

        y_i = r_{i+1} + gamma * \
            Q(s_{i+1}, argmax_a(Q(s_{i+1}, a; theta_i)); theta_i')

        See van Hasselt 2015 and the dqn_learn header.

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
        # Use primary network to generate qNext values for action
        # selection
        qNextPrimary = self.qNet.predict_on_batch(nextStates)
        # Get actions for next state
        nextActions = np.argmax(qNextPrimary, axis=1)
        # Use the target network and the actions chosen by the primary
        # network to get the qNext values
        qNext = self.tNet.predict_on_batch(nextStates)
        # Now get targetQ values as is done in dqn_learn
        qPred = self.qNet.predict_on_batch(states)
        qTarget = np.zeros(qPred.shape)
        doneInds = np.where(dones)
        nDoneInds = np.where(~dones)
        qTarget[doneInds, actions[doneInds]] = rewards[doneInds]
        qTarget[nDoneInds, actions[nDoneInds]] = (
            rewards[nDoneInds]
            + self.discountRate * qNext[nDoneInds, nextActions[nDoneInds]]
        )
        qTarget[qTarget == 0] = qPred[qTarget == 0]
        # Get abs error
        absError = tf.reduce_sum(tf.abs(qTarget - qPred), axis=1)
        # Update the network weights
        loss = self.qNet.train_on_batch(
            states, qTarget, sample_weight=isWeights
        )
        return STUFF 
