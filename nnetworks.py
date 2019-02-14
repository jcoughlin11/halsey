"""
Title:   nnetworks.py
Author:  Jared Coughlin
Date:    1/24/19
Purpose: Contains the class definitions of neural networks
Notes:
"""
import os
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras as keras
from tensorflow.train import AdamOptimizer

import nnutils as nu




#============================================
#              Deep-Q Network
#============================================
class DQNetwork():
    """
    This is the Deep-Q Network class. It defines the network architecture for training the
    agent using Deep-Q learning.

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
    def __init__(self, hyperparams, env, name, sess):
        # Initialize
        self.batchSize       = hyperparams['batch_size']
        self.callbacks       = None
        self.ckptFile        = hyperparams['ckpt_file']
        self.cropBot         = hyperparams['crop_bot']
        self.cropLeft        = hyperparams['crop_left']
        self.cropRight       = hyperparams['crop_right']
        self.cropTop         = hyperparams['crop_top']
        self.discountRate    = hyperparams['discount']
        self.env             = env
        self.epsDecayRate    = hyperparams['eps_decay_rate']
        self.epsilonStart    = hyperparams['epsilon_start']
        self.epsilonStop     = hyperparams['epsilon_stop']
        self.learningRate    = hyperparams['learning_rate']
        self.maxEpSteps      = hyperparams['max_steps']
        self.memSize         = hyperparams['memory_size']
        self.model           = None
        self.name            = name
        self.nEpisodes       = hyperparams['n_episodes']
        self.preTrainLen     = hyperparams['pretrain_len']
        self.renderFlag      = hyperparams['render_flag']
        self.restartTraining = hyperparams['restart_training']
        self.savePeriod      = hyperparams['save_period']
        self.saver           = None
        self.saveFilePath    = hyperparams['save_path']
        self.sess            = sess
        self.shrinkCols      = hyperparams['shrink_cols'] 
        self.shrinkRows      = hyperparams['shrink_rows']
        self.stackSize       = hyperparams['nstacked_frames']
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
        self.memory = nu.Memory(self.memSize,
                                self.preTrainLen,
                                self.env,
                                self.stackSize,
                                self.crop,
                                self.shrink)
        # Build the network
        self.model = self.build()

    #-----
    # Build
    #-----
    def build(self):
        """
        This function constructs the layers of the network.
        """
        # Make sure model has not already been initialized
        if self.model is not None:
            raise("Error, model aleady built!")
        else:
            with tf.variable_scope(self.name):
                # Placeholders (anything that needs to be fed from the outside)
                # Input. This is the stack of frames from the game
                self.inputs = tf.placeholder(tf.float32,
                                            shape=self.input_shape,
                                            name='inputs')
                # Actions. The action the agent chose. Used to get the predicted Q value
                self.actions = tf.placeholder(tf.float32,
                                              shape=[None, 1],
                                              name='actions')
                # Target Q. The max discounted future reward playing from next state
                # after taking chosen action. Determined by Bellmann equation.
                self.target_Q = tf.placeholder(tf.float32,
                                               shape=[None, self.env.action_space.n],
                                               name='target')

                # First convolutional layer
                self.conv1 = tf.layers.conv2d(inputs=self.inputs,
                    filters=32,
                    kernel_size=[8,8],
                    strides=[4,4],
                    padding='VALID',
                    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                    name='conv1')
                # First convolutional layer activation
                self.conv1_out = tf.nn.elu(self.conv1, name='conv1_out')

                # Second convolutional layer
                self.conv2 = tf.layers.conv2d(inputs=self.conv1_out,
                    filters=64,
                    kernel_size=[4,4],
                    strides=[2,2],
                    padding='VALID',
                    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                    name='conv2')
                # Second convolutional layer activation
                self.conv2_out = tf.nn.elu(self.conv2, name='conv2_out')

                # Third convolutional layer
                self.conv3 = tf.layers.conv2d(inputs=self.conv2_out,
                    filters=64,
                    kernel_size=[3,3],
                    strides=[2,2],
                    padding='VALID',
                    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                    name='conv3')
                # Third convolutional layer activation
                self.conv3_out = tf.nn.elu(self.conv3, name='conv3_out')

                # Flatten
                self.flatten = tf.contrib.layers.flatten(self.conv3_out)
                # FC layer
                self.fc = tf.layers.dense(inputs=self.flatten,
                    units=512,
                    activation=tf.nn.elu,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    name='fc1')
                # Output layer (FC)
                self.output = tf.layers.dense(inputs=self.fc,
                    units=self.env.action_space.n,
                    activation=None,
                    kernel_initializer=tf.contrib.layers.xavier_initializer())

                # Get the predicted Q value
                self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions))

                # Get the error (MSE)
                self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))

                # Optimizer
                self.optimizer = AdamOptimizer(self.learningRate).minimize(self.loss)
        self.saver = tf.train.Saver()

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
            self.saver.restore(self.sess, os.path.join(self.saveFilePath, 'si.ckpt'))
            train_params = nnutils.load_train_params(self.saveFilePath,
                                                    self.memory.maxlen)
            start_ep, decay_step, self.totalRewards, self.memory.buffer = train_params
        else:
            # Initialize tensorflow variables
            self.sess.run(tf.global_variables_initializer())
            # Set up the decay step for the epsilon-greedy search
            decay_step = 0
            start_ep = 0
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
                self.saver.save(self.sess, os.path.join(self.saveFilePath, 'si.ckpt'))
                nnutils.save_train_params(episode,
                                            decay_step,
                                            self.totalRewards,
                                            self.memory.buffer,
                                            self.saveFilePath)
            if early_abort is True:
                break
                
        # Save the final, trained model
        if early_abort is False:
            self.saver.save(self.sess, os.path.join(self.saveFilePath, 'si.ckpt'))
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
            # [0,5) gives us what we want
            action = np.random.randint(0, self.env.action_space.n)
        # Exploit
        else:
            # tf needs the batch size as part of the shape. See comment on
            # self.input_shape in the constructor
            state = state.reshape((1, state.shape[0], state.shape[1], state.shape[2]))
            Q_vals = self.sess.run(self.output,
                              feed_dict={self.inputs:state})
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
        action brings me to?" is given by the Bellmann equation.

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
        sample = self.memory.sample(self.batchSize)
        # There's no need to call sess.run self.batchSize times to get Q_target. It's not
        # wrong, but it is slow. It can be vectorized
        states      = np.array([s[0] for s in sample], ndmin=3)
        actions     = np.array([s[1] for s in sample])
        rewards     = np.array([s[2] for s in sample])
        next_states = np.array([s[3] for s in sample], ndmin=3)
        dones       = np.array([s[4] for s in sample])
        Q_next = self.sess.run(self.output, feed_dict={self.inputs : next_states})
        # Get an estimate of "how well can I do playing optimally from current state?"
        # This is our current Q table estimate
        Q_prediction = self.sess.run(self.output, feed_dict={self.inputs : states})
        # Loop over every experience in the sample in order to use them to update the Q
        # table
        for i in range(self.batchSize):
            # Begin by getting an estimate of the max discounted future reward we can
            # achieve by taking the chosen action and then playing optimally from the
            # state that action brings us to. This is called Q_target.
            # If the action brings us to a terminal state, then the max discounted future
            # reward we can achieve by playing optimally from the state our action
            # brought us to is just the reward given by the terminal state (since there
            # are no other states after it).
            if dones[i]:
                Q_target = rewards[i]
            # Otherwise, use the Bellmann equation
            else:
                Q_target = rewards[i] + self.discountRate * np.amax(Q_next[i])
            # Use Q_target to update the Q table. The values for the other actions stay
            # the same
            Q_prediction[i][actions[i]] = Q_target
        # Update network weights using the state as input and the updated Q values as
        # "labels." tf evaluates each element in the list it's given and returns one
        # value for each element. Note that this is called outside of the above for
        # loop in order to minimize the number of calls to sess.run that are needed
        # This requires that actions be reshaped to a column vector
        actions = actions.reshape((self.batchSize, 1))
        fd = {self.inputs : states,
                self.target_Q : Q_prediction,
                self.actions : actions}
        loss, _ = self.sess.run([self.loss, self.optimizer],
                            feed_dict=fd)
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
        self.saver.restore(self.sess, os.path.join(self.saveFilePath, 'si.ckpt'))
        # Play game
        for episode in range(1):
            episode_reward = 0
            state = self.env.reset()
            state, frame_stack = nu.stacK_frames(None,
                                                 state,
                                                 True,
                                                 self.stackSize,
                                                 self.crop,
                                                 self.shrink)
            # Loop until the agent fails
            while True:
                # Get what network thinks is the best action for current state
                Q_values = self.sess.run(self.output, feed_dict={self.inputs : state})
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
