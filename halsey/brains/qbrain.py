"""
Title: qbrain.py
Notes:
"""
import tensorflow as tf

from .base import BaseBrain


# ============================================
#                  QBrain
# ============================================
class QBrain(BaseBrain):
    """
    Implements the original deep-q learning method from DeepMind [1]_.

    Attributes
    ----------
    pass

    Methods
    -------
    learn(sample)
        Updates the network weights using the method described in [1]_.
    """

    # -----
    # constructor
    # -----
    def __init__(self, nets, params):
        """
        Doc string.
        """
        super().__init__(nets, params)
        self.discountRate = params["discountRate"]
        self.learningRate = params["learningRate"]

    # -----
    # learn
    # -----
    def learn(self, sample):
        """
        Updates the network weights using the method described in [1]_.

        In normal Q-learning we have a Q-table. The rows of this table
        are the states the agent can find itself in and the columns are
        the available actions. The numbers stored in each table entry
        represent the quality of taking the specified action in the
        specified state. These quality values are called Q-values.
        Larger Q-values indicate better quality.

        Before starting training, the agent has no knowledge about the
        quality of any action in any state. The Q-values are determined
        via self-play and actually observing how various actions in
        each state work out. Those that work out well have their
        Q-values increased and those that do not have their Q-values
        decreased.

        The Q-values are updated via the Bellmann Equation. This
        equation determines how well the agent can do if they play
        optimally (according to the Q-table) from the state their
        current action brought them to.

        The Bellmann Equation also applies a discount to those
        encounters that occur further in the future so that they have
        less influence over the current update than more immediate
        encounters do.

        Via this process, more play leads to more varied encounters and
        so better Q-value estimates. Better Q-value estimates allow the
        agent to better predict the quality of a state-action pair,
        which, in turn, leads to better Q-value estimates.

        For each action chosen in each encountered state, the
        environment provides the agent with a reward. The goal of the
        agent is to maximize the total rewards obtained over the course
        of a trajectory. This total reward is called the return. A
        trajectory is deemed good or bad based on its return.

        The main drawback of Q-learning is the fact that, for
        environments with a large number of states, actions, or both,
        maintaining a Q-table becomes untractable. This is where deep
        Q-learning (DQL) comes in. In this scenario, the full Q-table
        is approximated by a neural network, with updates to the
        network's weights replacing updates to the Q-table's Q-values.

        In the parlance of standard deep learning, the estimates
        provided by the Bellmann equation serve as the 'labels' for the
        weight update. These labels are usually referred to as
        'targets' in the RL literature.

        NOTE: This is heavily based on deepq_learner.py from OpenAI's
        baselines package.

        Parameters
        ----------
        sample : tuple
            An experience tuple containing the current state, chosen
            action, reward given by the environment for having chosen
            that action in that state, the state resulting from having
            taken the chosen action, and whether or not this next state
            is a terminal state or not. Each of these entries is a
            tensor whose first dimension is the batch size.

        Raises
        ------
        pass

        Returns
        -------
        Void
        """
        states, actions, rewards, nextStates, dones = sample
        with tf.GradientTape() as tape:
            # Get network's predictions for the quality of each action
            # in the current state
            qVals = self.nets[0](states, training=True)
            # qVals has shape (batchSize, nActions). However, the only
            # Q-values we need to update are those corresponding to the
            # chosen action. tf's graph helps us here because it keeps
            # track of each of these operations via tape. We use
            # tf.one_hot to make actions into a (batchSize, nActions)
            # sparse tensor, which, when multiplied by qVals, has only
            # non-zero values at the indices corresponding to the
            # chosen action
            nActions = qVals.shape[1]
            oneHot = qVals * tf.one_hot(actions, nActions, dtype=tf.float32)
            # We sum along axis 1 to get a (batchSize, 1) tensor whose
            # values are equal to the Q-values for the chosen action,
            # since all other values are zero
            qChosen = tf.reduce_sum(oneHot, 1)
            # Now we need to calculate the target Q-values. Start by
            # getting the network's guess for how well we can do by
            # playing optimally from the state our chosen action brings
            # us to. This is used in the Bellman equation
            qNext = self.nets[0](nextStates, training=True)
            qNextMax = tf.reduce_max(qNext, 1)
            # For those actions that resulted in a terminal state,
            # we need to assign just the reward given by the env
            # If the action did not result in a terminal state, we use
            # the Bellman equation. This can be done simultaneously via
            # a mask. dones has shape (batchSize, 1) and consists of
            # integer values, so we need to cast
            dones = tf.cast(dones, qNextMax.dtype)
            # This multiplication is element-by-element. Therefore, if
            # the next state is terminal, then done = 1, 1 - 1 = 0, and
            # therefore that entry of qNextMax is 0. If the next state
            # is not terminal, then done = 0, and qNextMax is untouched
            maskedVals = (1.0 - dones) * qNextMax
            # Now, if maskedVals is 0, those entries of targets is just
            # the given reward, and, if not, then it's the Bellman eq
            targets = rewards + self.discountRate * maskedVals
            # We don't want the derivative to depend on the targets,
            # and, since they come from variables watched by tape
            # (the network weights, via calling the network), we tell
            # tf to ignore them. These variables have to be watched
            # or else we can't take derivatives with respect to them
            loss = self.lossFunction(tf.stop_gradient(targets), qChosen)
        # Update the network weights by calculating the gradients and
        # applying them
        gradients = tape.gradient(loss, self.nets[0].trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.nets[0].trainable_variables)
        )

    # -----
    # get_state
    # -----
    def get_state(self):
        """
        Doc string.
        """
        return {}
