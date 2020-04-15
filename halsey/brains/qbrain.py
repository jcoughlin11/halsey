"""
Title: qbrain.py
Notes:
"""
import gin
import numpy as np
import tensorflow as tf

from .base import BaseBrain


# ============================================
#                  QBrain
# ============================================
@gin.configurable(blacklist=["nets"])
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
    def __init__(self, nets=None):
        """
        Doc string.
        """
        super().__init__(nets=nets)

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
        import pdb

        pdb.set_trace()
        states, actions, rewards, nextStates, dones = sample
        doneInds = np.where(dones)[0]
        nDoneInds = np.where(~dones)[0]
        # Get nextVals outside of tape so that this operation isn't
        # tracked by tape
        nextVals = self.nets[0](nextStates, training=True)
        with tf.GradientTape() as tape:
            # Getting predictions inside of tape causes this operation
            # to be tracked by tape because it involved watched variables
            # (the trainable variables of the network)
            predictions = self.nets[0](states, training=True)
            # Use the numpy data because it makes indexing SO much
            # easier
            # NOTE: If using the numpy data doesn't work when
            # decorating this function with @tf.function, then just use
            # a loop to update individual elements. tf.gather_nd and
            # tf.where can be used to get certain elements and their
            # indices in a tensor, but actually updating multiple
            # values at once like you can in numpy is just a pain and
            # requires masking and more convoluted code that just makes
            # things harder to read. or try tf.convert_to_tensor
            # Just to be safe, explicitly don't track this
            labels = tf.stop_gradient(predictions.numpy().copy())
            # For those state-action pairs that resulted in a terminal
            # state, set its output to just the reward
            # NOTE: Do I need stop_gradient here? I don't think so,
            # since this line doesn't involve anything trainable, so
            # it shouldn't affect the gradients. See next statement
            # This operation isn't tracked because it doesn't involve
            # any watched variables or variables made from watched operations
            labels[doneInds, actions[doneInds]] = rewards[doneInds]
            # For those state-action pairs that did not result in a
            # terminal state, use the Bellman equation to update
            # NOTE: Here, I think I need stop_gradient because this
            # line involves nextVals, which is related to trainable
            # variables. By that logic, shouldn't labels = preds.copy()
            # above use stop_gradient?
            # no longer need stop_gradient because this operation doesn't
            # involve any watched variables or any variables created from
            # tracked operations
            labels[nDoneInds, actions[nDoneInds]] = rewards[
                nDoneInds
            ] + self.discountRate * np.amax(
                tf.boolean_mask(nextVals, ~dones), axis=1
            )
            # The loss operation will be tracked because it involves
            # predictions, which is a variable created from a tracked
            # operation
            loss = self.lossFunction(labels, predictions)
        gradients = tape.gradient(loss, self.nets[0].trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.nets[0].trainable_variables)
        )
