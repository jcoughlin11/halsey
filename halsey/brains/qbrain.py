"""
Title: qbrain.py
Notes:
"""
import numpy as np
import tensorflow as tf

from .base import BaseBrain


# ============================================
#                   QBrain
# ============================================
class QBrain(BaseBrain):
    """
    Contains the learning method presented in Mnih et al. 2013.
    """

    # -----
    # learn
    # -----
    def learn(self):
        """
        The learning method by which the neural network weights are
        updated.

        NOTE: This is heavily based on deepq_learner.py from OpenAI's
        baselines package.

        NOTE: See version on development branch for in-depth comments.
        I wanto to add a more refined version of them here.
        """
        states, actions, rewards, nextStates, dones = self.memory.sample()
        with tf.GradientTape() as tape:
            beliefs = self.get_beliefs(states, actions)
            targets = self.get_targets(nextStates, dones, rewards)
            loss = self.nets[0].lossFunction(tf.stop_gradient(targets), beliefs)
        gradients = tape.gradient(loss, self.nets[0].trainable_variables)
        self.nets[0].optimizer.apply_gradients(
            zip(gradients, self.nets[0].trainable_variables)
        )

        # -----
        # get_beliefs
        # -----
        def get_beliefs(self, states, actions):
            """
            Gets the network's current guess for the Q-value of each
            action in the given state.
            """
            qVals = self.nets[0](states, training=True)
            nActions = qVals.shape[1]
            oneHot = qVals * tf.one_hot(actions, nActions, dtype=tf.float32)
            qChosen = tf.reduce_sum(oneHot, 1)
            return qChosen

        # -----
        # get_targets
        # -----
        def get_targets(self, nextStates, dones, rewards):
            """
            Bootstraps the "true" Q-values to compare to the beliefs
            in the loss function.
            """
            qNext = self.nets[0](nextStates, training=True)
            qNextMax = tf.reduce_max(qNext, 1)
            dones = tf.cast(dones, qNextMax.dtype)
            maskedVals = (1.0 - dones) * qNextMax
            targets = rewards + self.discountRate * maskedVals
            return targets

    # -----
    # predict
    # -----
    def predict(self, state):
        """
        Uses the current knowledge of the neural network(s) to select
        what it thinks is the best action for the current situation.
        """
        predictions = self.nets[0](state, training=False)
        return np.argmax(predictions.numpy())
