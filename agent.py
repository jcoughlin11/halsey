"""
Title:   nnetworks.py
Author:  Jared Coughlin
Date:    1/24/19
Purpose: Contains the class definitions of neural networks
Notes:
"""
import os
import subprocess32
import sys
import time

import numpy as np
import tensorflow as tf

import nnetworks as nw
import nnutils as nu




#============================================
#                    Agent
#============================================
class Agent():
    """
    This is the Agent class. It contains all of the methods for using various RL
    techniques in order to learn to play games based on just the pixels on the game
    screen.

    Parameters:
    -----------
        hyperparams: dict
            A dictionary containing the relevant hyperparameters. See read_hyperparams
            in nnutils.py

        env : gym environment
            This is the game's environment, created by gym, that contains all of the
            relevant details about the game

        name : string
            The name that tensorflow assigns to the variable namespace

        sess : tf.Session()
            The tensorflow session used to evaluate tensors. Originally, I created new
            sessions in the functions that needed to evaluate tensors, such as self.learn.
            However, tf complained because the variables had not been initialzed for that
            session, so by passing it, we create a persistent session that spans the scope
            of the whole class.
    """
    #-----
    # Constructor
    #-----
    def __init__(self, hyperparams, env, sess):
        # Initialize
        self.batchSize       = hyperparams['batch_size']
        self.callbacks       = None
        self.ckptFile        = hyperparams['ckpt_file']
        self.cropBot         = hyperparams['crop_bot']
        self.cropLeft        = hyperparams['crop_left']
        self.cropRight       = hyperparams['crop_right']
        self.cropTop         = hyperparams['crop_top']
        self.discountRate    = hyperparams['discount']
        self.doubleDQN       = hyperparams['double_dqn']
        self.env             = env
        self.epsDecayRate    = hyperparams['eps_decay_rate']
        self.epsilonStart    = hyperparams['epsilon_start']
        self.epsilonStop     = hyperparams['epsilon_stop']
        self.fixedQ          = hyperparams['fixed_Q']
        self.fixedQSteps     = hyperparams['fixed_Q_steps']
        self.learningRate    = hyperparams['learning_rate']
        self.maxEpSteps      = hyperparams['max_steps']
        self.memSize         = hyperparams['memory_size']
        self.nEpisodes       = hyperparams['n_episodes']
        self.per             = hyperparams['per']
        self.perA            = hyperparams['per_a']
        self.perB            = hyperparams['per_b']
        self.perBAnneal      = hyperparams['per_b_anneal']
        self.perE            = hyperparams['per_e']
        self.preTrainLen     = hyperparams['pretrain_len']
        self.qNet            = None
        self.renderFlag      = hyperparams['render_flag']
        self.restartTraining = hyperparams['restart_training']
        self.savePeriod      = hyperparams['save_period']
        self.saver           = None
        self.saveFilePath    = hyperparams['save_path']
        self.sess            = sess
        self.shrinkCols      = hyperparams['shrink_cols'] 
        self.shrinkRows      = hyperparams['shrink_rows']
        self.stackSize       = hyperparams['nstacked_frames']
        self.targetQNet      = None
        self.totalRewards    = []
        # Seed rng
        np.random.seed(int(time.time()))
        # Set up tuples for preprocessed frame sizes
        self.crop = (self.cropTop, self.cropBot, self.cropLeft, self.cropRight)
        self.shrink = (self.shrinkRows, self.shrinkCols)
        # Use None for batch size in shape b/c in predict action we give it 1 state, but
        # in self.learn(), we give it self.batchSize states all at once. This vectorizes
        # the feedforward pass and makes it much faster than going one experience tuple
        # at a time, which is what I did in the keras version
        self.input_shape = (None,
                            self.shrinkRows,
                            self.shrinkCols,
                            self.stackSize)
        # Set up memory
        if self.per == 1:
            perParams = [self.perA,
                        self.perB,
                        self.perBAnneal,
                        self.perE
                        ]
            self.memory = nu.Memory(self.memSize,
                                    self.preTrainLen,
                                    self.env,
                                    self.stackSize,
                                    self.crop,
                                    self.shrink,
                                    perParams)
        else:
            self.memory = nu.Memory(self.memSize,
                                    self.preTrainLen,
                                    self.env,
                                    self.stackSize,
                                    self.crop,
                                    self.shrink)
        # Build the network
        self.qNet = nw.DQN('qnet', hyperparams['architecture'], self.input_shape,
                            self.env.action_space.n, self.learningRate)
        # If applicable, build the second network for use with the fixed-Q technique.
        # This also creates the second network for double dqn since fixed-Q is required
        # for that
        if self.fixedQ == 1:
            self.targetQNet = nw.DQN('targetQNet', hyperparams['architecture'],
                                    self.input_shape, self.env.action_space.n,
                                    self.learningRate)

    #----
    # Train
    #----
    def train(self, restart=True):
        """
        This function trains the agent to play the game.

        Parrameters:
        ------------
            restart : bool
                If true, start training from the beginning. If false, load a saved model and
                continue where training last left off.

        Returns:
        --------
            None
        """
        early_abort = False
        # Initialize the tensorflow session (uses default graph)
        # See if we need to load a saved model to continue training
        if restart is False:
            self.qNet.saver.restore(self.sess, os.path.join(self.saveFilePath,
                                    self.ckptFile + '.ckpt'))
            if self.fixedQ == 1:
                self.targetQNet.saver.restore(self.sess, os.path.join(self.saveFilePath,
                                            self.ckptFile + '-target.ckpt'))
            train_params = nu.load_train_params(self.saveFilePath,
                                                    self.memory.max_size)
            start_ep,\
            decay_step,\
            self.totalRewards,\
            self.memory.buffer,\
            fixed_Q_step = train_params
        else:
            # Initialize tensorflow variables
            self.sess.run(tf.global_variables_initializer())
            # Now that the global variable initializer has been run, we can copy the
            # weights from qNet to targetQNet if using fixed-Q . If I've
            # done this right, this should make it so both networks now have the same
            # weights when training starts
            if self.fixedQ == 1:
                updateTarget = self.update_target_graph()
                self.sess.run(updateTarget)
            # Set up the decay step for the epsilon-greedy search
            decay_step = 0
            start_ep = 0
            fixed_Q_step = 0
        # Loop over desired number of training episodes
        for episode in range(start_ep, self.nEpisodes):
            print('Episode: %d / %d' % (episode + 1, self.nEpisodes))
            # Reset time spent on current episode
            step = 0
            # Track the rewards for the episode
            episode_rewards = []
            # Reset the environment
            state = self.env.reset()
            # Stack and process initial state. State is returned as a tensor of shape
            # (frame_rows, frame_cols, stack_size). frame_stack is the same data, but
            # in the form of a deque.
            state, frame_stack = nu.stack_frames(None,
                                                 state,
                                                 True,
                                                 self.stackSize,
                                                 self.crop,
                                                 self.shrink)
            # Loop over the max amount of time the agent gets per episode
            while step < self.maxEpSteps:
                print('Step: %d / %d' % (step, self.maxEpSteps), end='\r')
                # Increase step counters
                step += 1
                decay_step += 1
                fixed_Q_step += 1
                # Choose an action
                action = self.choose_action(state, decay_step)
                # Perform action
                next_state, reward, done, _ = self.env.step(action)
                # Track the reward
                episode_rewards.append(reward)
                # Add the next state to the stack of frames
                next_state, frame_stack = nu.stack_frames(frame_stack,
                                                          next_state,
                                                          False,
                                                          self.stackSize,
                                                          self.crop,
                                                          self.shrink)
                # Save experience
                experience = (state, action, reward, next_state, done)
                self.memory.add(experience)
                # Learn from the experience
                loss = self.learn()
                # Update the targetQNet if applicable
                if self.fixedQ == 1:
                    if fixed_Q_step > self.fixedQSteps:
                        fixed_Q_step = 0
                        updateTarget = self.update_target_graph()
                        self.sess.run(updateTarget)
                # Set up for next episode if we're in a terminal state
                if done:
                    # Get total reward for episode
                    tot_reward = np.sum(episode_rewards)
                    # Save total episode reward
                    self.totalRewards.append(tot_reward)
                    # Print info to screen
                    print('Episode: {}\n'.format(episode),
                            'Total Reward for episode: {}\n'.format(tot_reward),
                            'Training loss: {:.4f}'.format(loss))
                    # See if training is being ended early
                    if os.path.isfile(os.path.join(os.getcwd(), 'stop')):
                        early_abort = True
                        subprocess32.call(['rm', os.path.join(os.getcwd(), 'stop')])
                        break
                    break
                # Set up for next step
                else:
                    state = next_state
                    # See if training is being ended early
                    if os.path.isfile(os.path.join(os.getcwd(), 'stop')):
                        early_abort = True
                        subprocess32.call(['rm', os.path.join(os.getcwd(), 'stop')])
                        break
            # If we finish because of a time out (i.e, we reach the max number of
            # steps, then we want to get the total episode reward
            if (step == self.maxEpSteps) and (done is False):
                self.totalRewards.append(np.sum(episode_rewards))
            # Save the model
            if ((episode % self.savePeriod == 0) and (done is True)) or\
                ((episode % self.savePeriod == 0) and (step == self.maxEpSteps)):
                # Make sure the directory exists
                if not os.path.exists(self.saveFilePath):
                    os.mkdir(self.saveFilePath)
                self.qNet.saver.save(self.sess, os.path.join(self.saveFilePath,
                                    self.ckptFile + '.ckpt'))
                if self.fixedQ == 1:
                    self.targetQNet.saver.save(self.sess,
                                            os.path.join(self.saveFilePath,
                                            self.ckptFile + '-target.ckpt'))
                nu.save_train_params(decay_step,
                                            self.totalRewards,
                                            self.memory.buffer,
                                            self.saveFilePath,
                                            fixed_Q_step)
            if early_abort is True:
                break
                
        # Save the final, trained model
        if early_abort is False:
            if not os.path.exists(self.saveFilePath):
                os.mkdir(self.saveFilePath)
            self.qNet.saver.save(self.sess, os.path.join(self.saveFilePath,
                                self.ckptFile + '.ckpt'))
            if self.fixedQ == 1:
                self.targetQNet.saver.save(self.sess,
                                        os.path.join(self.saveFilePath,
                                        self.ckptFile + '-target.ckpt'))
        if early_abort is True:
            sys.exit()

    #-----
    # Choose Action
    #-----
    def choose_action(self, state, decay_step):
        """
        This function uses the current state and the agent's current knowledge in
        order to choose an action. It employs the epsilon greedy strategy to handle
        exploration vs. exploitation.

        Parameters:
        ------------
            state : ndarray
                The tensor of stacked frames

            decay_step : int
                The overall training step. This is used to cause the agent to favor
                exploitation in the long-run

        Returns:
        --------
            action : gym action
                The agent's choice of what to do based on the current state
        """
        # Choose a random number from uniform distribution between 0 and 1. This is
        # the probability that we exploit the knowledge we already have
        exploit_prob = np.random.random()
        # Get the explore probability. This probability decays over time (but stops
        # at eps_stop so we always have some chance of trying something new) as the
        # agent learns
        explore_prob = self.epsilonStop +\
                        (self.epsilonStart - self.epsilonStop) *\
                        np.exp(-self.epsDecayRate * decay_step)
        # Explore
        if explore_prob >= exploit_prob:
            # Choose randomly. randint selects integers in the range [a,b). So, if
            # there are 5 possible actions, we want to select 0, 1, 2, 3, or 4.
            # [0,5) gives us what we want. Can just be action_space.sample?
            action = np.random.randint(0, self.env.action_space.n)
        # Exploit
        else:
            # tf needs the batch size as part of the shape. See comment on
            # self.input_shape in the constructor
            state = state.reshape((1, state.shape[0], state.shape[1], state.shape[2]))
            Q_vals = self.sess.run(self.qNet.output,
                              feed_dict={self.qNet.inputs:state})
            # Choose the one with the highest Q value
            action = np.argmax(Q_vals)
        return action

    #-----
    # Learn
    #-----
    def learn(self):
        """
        This function trains the network by sampling from the experience (memory)
        buffer and uses those experiences as the training set.

        Our network is our Q table. It tells us, for every state, what the max discounted
        future reward is for each action. Our policy is to select the action with the
        largest max discounted future reward at each state. The network is an 
        approximation to the full Q table, since the full thing is too large to be
        represented exactly.

        We start by not knowing anything about the max discounted future reward for any
        (state, action) pair. By playing, the environment (game) gives us rewards (either
        positive or negative) for each action. This allows us to slowly fill in the Q
        table.

        However, how do we train the network? That is, when we take an action, we are
        given a reward, but is that the max reward for that state? Further, how do we
        know what max discounted future reward each (state, action) pair gives rise to?
        Additionally, at least initially, an action that gives rise to a large short-term
        reward might not actually be the best long-term decision, which is why we need an
        estimate of the "how well can I do playing optimally from the state my action
        brings me to?" (i.e., the Bellmann equation) that we can compare to what our
        network currently thinks the max discounted future reward is.

        The answer is to compare how well we would do continuing to play from the state
        our current action brings us to with how well we would do by playing optimally
        from our current state (where the optimal action is not necessarily the action we
        chose). The idea is that, if we're playing optimally, then the two should match.
        The trouble is that we don't actually know the Q table ahead of time, so we have
        to use both estimates, along with the rewards received from the game, in order to
        learn. This makes DRL a hybrid cross between supervised and unsupervised learning.

        The estimate of the "how well can I do playing optimally from the state my current
        action brings me to?" is given by the Bellmann equation. These are the target Q
        values.

        Begin by getting an estimate of the max discounted future reward we can
        achieve by taking the chosen action and then playing optimally from the
        state that action brings us to. This is called Q_target.

        If the action brings us to a terminal state, then the max discounted future
        reward we can achieve by playing optimally from the state our action
        brought us to is just the reward given by the terminal state (since there
        are no other states after it).

        If the current action for the current state does not result in a
        terminal state, then we need to make an estimate about what the max
        discounted future reward will be (Bellmann equation)

        We then ask, "How well can I do if I play optimally from the current state onwards
        (where the optimal action is not necessarily the chosen action)?" This is the value
        we get from our Q table (network) Q_prediction. These values are updated constantly
        as the agent gets reward information from the environment.

        This way, information about rewards lets us get better Q_target values, which
        improve our Q_preidctions, which, along with more information about rewards,
        allows us to improve Q_target, and so on, forming a feedback loop.

        Returns:
        -------
            loss : float
                The value of the loss function for the current training run
        """
        # Get sample of experiences
        if self.per == 1:
            treeInds, sample, isWeights = self.memory.sample(self.batchSize)
        else:
            sample = self.memory.sample(self.batchSize)
        # There's no need to call sess.run self.batchSize times to get Q_target. It's not
        # wrong, but it is slow. It can be vectorized
        states      = np.array([s[0] for s in sample], ndmin=3)
        actions     = np.array([s[1] for s in sample])
        rewards     = np.array([s[2] for s in sample])
        next_states = np.array([s[3] for s in sample], ndmin=3)
        dones       = np.array([s[4] for s in sample])
        # Double dqn
        if self.doubleDQN == 1:
            # Double dqn attempts to deal with the following issue: when we choose the
            # action that gives rise to the highest Q value for the next state, how do
            # we know that that's actually the best action? Since we're learning from
            # experience, our estimated Q values depend on which actions have been tried
            # and which neighboring states have been visited. As such, double dqn
            # separates out the estimate of the Q value and the determination of the best
            # action to take at the next state. We use our primary network to choose an
            # action for the next state and then pass that action to our target network,
            # which handles calculating the target Q value. That is:
            # Q_target(s,a) = r(s,a) + gamma * Q_target(s', argmax(Q(s',a)))
            Q_next = self.sess.run(self.qNet.output,
                                    feed_dict={self.qNet.inputs : next_states})
            Q_target_next = self.sess.run(self.targetQNet.output,
                                        feed_dict={self.targetQNet.inputs : next_states})
        # Fixed Q
        elif self.fixedQ == 1:
            Q_next = self.sess.run(self.targetQNet.output,
                                    feed_dict={self.targetQNet.inputs : next_states})
        # Standard dqn
        else:
            Q_next = self.sess.run(self.qNet.output,
                                    feed_dict={self.qNet.inputs : next_states})
        # Loop over every experience in the sample in order to use them to update the Q
        # table
        targetQ = np.zeros(self.batchSize)
        for i in range(self.batchSize):
            # Begin by getting an estimate of the max discounted future reward we can
            # achieve by taking the chosen action and then playing optimally from the
            # state that action brings us to. This is called Q_target.
            # If the action brings us to a terminal state, then the max discounted future
            # reward we can achieve by playing optimally from the state our action
            # brought us to is just the reward given by the terminal state (since there
            # are no other states after it).
            if dones[i]:
                targetQ[i] = rewards[i]
            # Otherwise, use the Bellmann equation
            else:
                # If using double dqn, this is where we apply the Q value for the action
                # chosen by our target network (see eq in double_dqn if block above)
                if self.doubleDQN == 1:
                    action = np.argmax(Q_next[i])
                    targetQ[i] = rewards[i] + self.discountRate * Q_target_next[i][action]
                # Normal DQL method: uses action that gives max Q value for next state
                else:
                    targetQ[i] = rewards[i] + self.discountRate * np.amax(Q_next[i])
        # Update network weights using the state as input and the updated Q values as
        # "labels." tf evaluates each element in the list it's given and returns one
        # value for each element. Note that this is called outside of the above for
        # loop in order to minimize the number of calls to sess.run that are needed
        # This requires that actions be reshaped to a column vector
        actions = actions.reshape((self.batchSize, 1))
        if self.per == 1:
            fd = {self.qNet.inputs : states,
                    self.qNet.target_Q : targetQ,
                    self.qNet.actions : actions,
                    self.qNet.isWeights : isWeights}
            _, loss, errors = self.sess.run([self.qNet.optimizer, self.qNet.loss,
                                            self.qNet.errors], feed_dict=fd)
            # Update the priorities in the tree
            self.memory.update(treeInds, errors)
        else:
            fd = {self.qNet.inputs : states,
                    self.qNet.target_Q : targetQ,
                    self.qNet.actions : actions}
            loss, _ = self.sess.run([self.qNet.loss, self.qNet.optimizer], feed_dict=fd)
        return loss

    #-----
    # Test
    #-----
    def test(self, render_flag):
        """
        This function acutally has the agent play the game using what it's learned so we
        can see how well it does.

        Parameters:
        -----------
            render_flag : int
                If 1, then we have gym draw the game to the screen for us so we can see
                what the agent is doing. Otherwise, nothing is drawn and only the final
                info about the agent's performance is given.

        Returns:
        --------
            None
        """
        # Load model
        self.qNet.saver.restore(self.sess, os.path.join(self.saveFilePath,
                                self.ckptFilei + '.ckpt'))
        # Play game
        for episode in range(1):
            episode_reward = 0
            state = self.env.reset()
            state, frame_stack = nu.stack_frames(None,
                                                 state,
                                                 True,
                                                 self.stackSize,
                                                 self.crop,
                                                 self.shrink)
            # Loop until the agent fails
            while True:
                # tf needs the batch size as part of the shape. See comment on
                # self.input_shape in the constructor
                state = state.reshape((1, state.shape[0], state.shape[1], state.shape[2]))
                # Get what network thinks is the best action for current state
                Q_values = self.sess.run(self.qNet.output,
                                        feed_dict={self.qNet.inputs : state})
                action = np.argmax(Q_values)
                # Take chosen action
                next_state, reward, done, _ = self.env.step(action)
                if render_flag:
                    self.env.render()
                episode_reward += reward
                # Check for terminal state
                if done:
                    print('Total score: %d' % (episode_reward))
                    break
                # If not done, get the next state and add it to the stack of frames
                else:
                    next_state, frame_stack = nu.stack_frames(frame_stack,
                                                              next_state,
                                                              False,
                                                              self.stackSize,
                                                              self.crop,
                                                              self.shrink)
                    state = next_state
        # Close the environment when we're done
        self.env.close()

    #-----
    # update_target_graph
    #-----
    def update_target_graph(self):
        """
        This function handles copying over the parameters from qNet to targetQNet in
        order to update the target network when using fixed-Q. See notes
        from 3/6/19.

        Parameters:
        -----------
            None

        Returns:
        --------
            opHolder : list
                This is a list of tensorflow variables from qNet to be copied over to
                targetQNet
        """
        # Get variables to copy
        fromVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'qNet')
        # Get variables to be updated
        toVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'targetQNet')
        # Initialize the list to hold variables to be updated and the values to update
        # them with
        opHolder = []
        # Fill in the list
        for fromVar, toVar in zip(fromVars, toVars):
            opHolder.append(toVar.assign(fromVar))
        return opHolder
