import numpy as np
import tensorflow as tf

from .base import BaseBrain


# ============================================
#                   QBrain
# ============================================
class QBrain(BaseBrain):
    """
    Contains the learning method presented in [Mnih et al. 2013][1].

    [1]: https://arxiv.org/abs/1312.5602
    """

    # -----
    # learn
    # -----
    def learn(self):
        """
        Compares network's beliefs with estimated "correct" answers as
        determined by the Bellman equation.

        !!! note "Based on deepq_learner.py from OpenAI's baselines."
        """
        states, actions, rewards, nextStates, dones = self.memory.sample()
        # GradientTape tracks operations so gradients can be found
        with tf.GradientTape() as tape:
            beliefs = self.get_beliefs(states, actions)
            targets = self.get_targets(nextStates, dones, rewards)
            # When taking a gradient, we don't want to treat targets
            # as functions of the weights
            loss = self.nets[0].loss(tf.stop_gradient(targets), beliefs)
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

        Arguments:
            states (np.ndarray): Batch of the current states the agent
                finds itself in.
            actions (np.ndarray): Batch of integer values representing
                the actions chosen by the agent in the given state.

        Returns:
            qChosen (tf.EagerTensor): The network's predictions for the
                Q-values of the current actions in the given states.
        """
        qVals = self.nets[0](states, training=True)
        nActions = qVals.shape[1]
        # We only need to change the Q-values for the chosen actions,
        # so we use a one-hot to vectorize the calculation while
        # simultaneously keeping the non-chosen indices untouched
        oneHot = qVals * tf.one_hot(actions, nActions, dtype=tf.float32)
        qChosen = tf.reduce_sum(oneHot, 1)
        return qChosen

    # -----
    # get_targets
    # -----
    def get_targets(self, nextStates, dones, rewards):
        """
        Uses the Bellman equation to estimate the "true" Q-values.

        Arguments:
            nextStates (np.ndarray): Batch of states resulting from
                taking the chosen action in the current state.
            dones (np.ndarray): Batch of boolean values representing
                whether or not the nextStates are terminal.
            rewards (np.ndarray): Batch of numerical rewards provided
                to the agent by the game for having taken the chosen
                action in the current state.

        Returns:
            targets (tf.EagerTensor): Estimates of the "true" Q-values
                of each action for the current state as estimated by
                the Bellman equation.
        """
        qNext = self.nets[0](nextStates, training=True)
        qNextMax = tf.reduce_max(qNext, 1)
        dones = tf.cast(dones, qNextMax.dtype)
        # The mask lets us apply both parts of the learning algo
        # at the same time: if the state is terminal, only the given
        # reward is used. If it's not terminal, the estimate from the
        # Bellman equation is applied
        maskedVals = (1.0 - dones) * qNextMax
        targets = rewards + self.params["discountRate"] * maskedVals
        return targets

    # -----
    # predict
    # -----
    def predict(self, state):
        """
        Uses the neural network's current knowledge to select what it
        thinks is the best action for the current situation.

        Arguments:
            state (np.ndarray): Array comprising the group of stacked
                game frames.

        Returns:
            int: Integer corresponding to the chosen action.
        """
        # Batch size is always required, even if it's just 1
        state = tf.expand_dims(state, 0)
        predictions = self.nets[0](state, training=False)
        return np.argmax(predictions.numpy())
