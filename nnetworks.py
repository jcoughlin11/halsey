"""
Title:   nnetworks.py
Author:  Jared Coughlin
Date:    1/24/19
Purpose: Contains the class definitions of neural networks
Notes:
"""
import time

import numpy as np
import pyprind
import tensorflow as tf
from tensorflow import keras as keras
from tensorflow.keras.optimizers import Adam

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
    """
    #-----
    # Constructor
    #-----
    def __init__(self, hyperparams, env):
        # Initialize
        self.batchSize    = hyperparams['batch_size']
        self.callbacks    = None
        self.cropBot      = hyperparams['crop_bot']
        self.cropLeft     = hyperparams['crop_left']
        self.cropRight    = hyperparams['crop_right']
        self.cropTop      = hyperparams['crop_top']
        self.discountRate = hyperparams['discount']
        self.env          = env
        self.epsDecayRate = hyperparams['eps_decay_rate']
        self.epsilonStart = hyperparams['epsilon_start']
        self.epsilonStop  = hyperparams['epsilon_stop']
        self.learningRate = hyperparams['learning_rate']
        self.maxEpSteps   = hyperparams['max_steps']
        self.memSize      = hyperparams['memory_size']
        self.model        = None
        self.nEpisodes    = hyperparams['n_episodes']
        self.preTrainLen  = hyperparams['pretrain_len']
        self.renderFlag   = hyperparams['render_flag']
        self.shrinkCols   = hyperparams['shrink_cols'] 
        self.shrinkRows   = hyperparams['shrink_rows']
        self.stackSize    = hyperparams['nstacked_frames']
        self.totalRewards = []
        # Set up callbacks
        self.callbacks = nu.set_up_callbacks()
        # Seed rng
        np.random.seed(int(time.time()))
        # Set up tuples for preprocessed frame sizes
        self.crop = (self.cropTop, self.cropBot, self.cropLeft, self.cropRight)
        self.shrink = (self.shrinkRows, self.shrinkCols)
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
            # Set up network architecture
            model = keras.Sequential()
            # First convolutional layer. Keras expects the number of chanels to be
            # given as a part of the input shape. Grayscaling the images removes this
            # channel info, so here it gets added back
            in_shape = (self.shrinkRows,
                        self.shrinkCols,
                        self.stackSize)
            model.add(keras.layers.Conv2D(input_shape = in_shape,
                                            filters=32,
                                            kernel_size=[8,8],
                                            strides=[4,4],
                                            activation=tf.nn.elu,
                                            data_format='channels_last'
                                         ))
            # Second convolutional layer
            model.add(keras.layers.Conv2D(filters=64,
                                            kernel_size=[4,4],
                                            strides=[2,2],
                                            activation=tf.nn.elu
                                         ))
            # Third convolutional layer
            model.add(keras.layers.Conv2D(filters=64,
                                            kernel_size=[3,3],
                                            strides=[2,2],
                                            activation=tf.nn.elu
                                         ))
            # Flatten later
            model.add(keras.layers.Flatten())
            # Fully-Connected (FC) layer
            model.add(keras.layers.Dense(units=512,
                                        activation=tf.nn.elu
                                        ))
            # Output layer
            model.add(keras.layers.Dense(units=self.env.action_space.n))
            # Compile
            model.compile(loss='mse',
                        optimizer=Adam(lr=self.learningRate)
                        )
            return model

    #----
    # Train
    #----
    def train(self):
        """
        This function trains the agent to play the game.

        Returns:
        --------
            None
        """
        # Set up the decay step for the epsilon-greedy search
        decay_step = 0
        pbar = pyprind.ProgBar(self.nEpisodes)
        for episode in range(self.nEpisodes):
            #pbar.update()
            print('\n')
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
                    break
                # Set up for next step
                else:
                    state = next_state

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
            Q_vals = self.model.predict(state)
            # Choose the one with the highest Q value. I tested this and the predict
            # method returns an array of shape (1, nfeatures). Therefore, to actually
            # get at the values, we have to use [0] to access the values in the row
            action = np.argmax(Q_vals[0])
        return action

    #-----
    # Learn
    #-----
    def learn(self):
        """
        This function trains the network by sampling from the experience (memory)
        buffer and uses those experiences as the training set.

        Returns:
        --------
            loss : float
                The value of the loss function for the current training run
        """
        # Get sample
        sample = self.memory.sample(self.batchSize)

        # Loop over each experience tuple in the sample
        for state, action, reward, next_state, done in sample:
            # If the action taken in the current state results in a terminal state,
            # then there's nothing to update since the game is done. We just leave
            # the target Q value as whatever reward is given for getting to the
            # terminal state
            if done:
                Q_target = reward
            # If the current action for the current state does not result in a
            # terminal state, then we need to make an estimate about what the max
            # discounted future reward is. This means we need to estimate what our
            # maximum discounted future reward is starting at the state the current
            # action brings us to. This is done via the Bellmann equation. That is,
            # we're asking, "If I play optimally from where my current action brought
            # me, how well can I do?" This is Q_target. We then ask, "How well can I
            # do if I play optimally from the current state onwards (where the optimal
            # action is not necessarily the chosen action)?" The idea is that the
            # "how well can I do playing optimally from the next state onwards" should
            # match the "how well can I do playing from the current state onwards"
            # because that means our chosen action was optimal. We use the difference
            # between the two, along with information about the rewards given, to
            # update our Q table (network).
            else:
                # Get the "how well can I do playing optimally from next state on"
                # The [0] is needed becauses of the shape of predict's return is (1,N)
                Q_target = reward +\
                            self.discountRate *\
                            np.amax(self.model.predict(next_state)[0])
            # Get the "how well can I do playing from current state on"
            Q_pred = self.model.predict(state)
            # Now, we update our Q table. This is done by setting the entry for
            # the current action and current state to the "how well can I do
            # playing optimally from next state on?" That is, we're telling the
            # network, "if you take the current action in the current state, this
            # is how well you can do." The Q values for all the other actions
            # remain untouched
            Q_pred[0][action] = Q_target
            # Now we update the weights for the network in order to reflect the
            # changes we made to the Q table
            hist = self.model.fit(state, Q_pred, epochs=1, verbose=0,
                                callbacks=self.callbacks)
        return hist.history['loss']

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
                Q_values = self.model.predict(state)
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
