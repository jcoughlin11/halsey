"""
Title:   agent.py
Author:  Jared Coughlin
Date:    1/24/19
Purpose: Contains the Agent class, which is the object that learns
Notes:
"""
import os
import subprocess32
import sys
import time

import numpy as np
import tensorflow as tf

import memory as mem
import nnetworks as nw
import nnio as io


#============================================
#                    Agent
#============================================
class Agent:
    """
    Contains all of the methods for using various RL techniques in
    order to learn to play games based on just the pixels on the game
    screen.

    Attributes:
    -----------
        pass

    Methods:
    --------
        pass
    """

    # -----
    # Constructor
    # -----
    def __init__(self, hyperparams, env):
        """
        Parameters:
        -----------
            hyperparams: dict
                A dictionary containing the relevant hyperparameters.
                See read_hyperparams in nnutils.py.

            env : gym.core.Env
                This is the game's environment, created by gym, that
                contains all of the relevant details about the game.

        Raises:
        -------
            pass

        Returns:
        --------
            pass
        """
        # Initialize
        self.batchSize       = hyperparams["batch_size"]
        self.ckptFile        = hyperparams["ckpt_file"]
        self.cropBot         = hyperparams["crop_bot"]
        self.cropLeft        = hyperparams["crop_left"]
        self.cropRight       = hyperparams["crop_right"]
        self.cropTop         = hyperparams["crop_top"]
        self.discountRate    = hyperparams["discount"]
        self.enableDoubleDQN = hyperparams["enable_double_dqn"]
        self.enableFixedQ    = hyperparams["enable_fixed_Q"]
        self.enablePer       = hyperparams["enable_per"]
        self.env             = env
        self.epsDecayRate    = hyperparams["eps_decay_rate"]
        self.epsilonStart    = hyperparams["epsilon_start"]
        self.epsilonStop     = hyperparams["epsilon_stop"]
        self.fixedQSteps     = hyperparams["fixed_Q_steps"]
        self.learningRate    = hyperparams["learning_rate"]
        self.maxEpSteps      = hyperparams["max_episode_steps"]
        self.memSize         = hyperparams["memory_size"]
        self.nEpisodes       = hyperparams["n_episodes"]
        self.perA            = hyperparams["per_a"]
        self.perB            = hyperparams["per_b"]
        self.perBAnneal      = hyperparams["per_b_anneal"]
        self.perE            = hyperparams["per_e"]
        self.preTrainEpLen   = hyperparams["pretrain_max_ep_len"]
        self.preTrainLen     = hyperparams["pretrain_len"]
        self.qNet            = None
        self.renderFlag      = hyperparams["render_flag"]
        self.restartTraining = hyperparams["restart_training"]
        self.savePeriod      = hyperparams["save_period"]
        self.saveFilePath    = hyperparams["save_path"]
        self.shrinkCols      = hyperparams["shrink_cols"]
        self.shrinkRows      = hyperparams["shrink_rows"]
        self.stackSize       = hyperparams["n_stacked_frames"]
        self.targetQNet      = None
        self.totalRewards    = []
        self.traceLen        = hyperparams["trace_len"]
        # Seed the rng
        np.random.seed(int(time.time()))
        # Set up tuples for preprocessed frame sizes
        self.crop = (self.cropTop, self.cropBot, self.cropLeft, self.cropRight)
        self.shrink = (self.shrinkRows, self.shrinkCols)
        # Set the size of the input frame stack
        self.input_shape = (
            self.shrinkRows,
            self.shrinkCols,
            self.stackSize,
        )
        # Set up memory
        if self.enablePer:
            perParams = [self.perA, self.perB, self.perBAnneal, self.perE]
            self.memory = mem.PriorityMemory(
                self.memSize, self.preTrainLen, perParams
            )
        elif hyperparams["architecture"] == "rnn1":
            self.memory = mem.EpisodeMemory(
                self.memSize, self.preTrainLen, self.preTrainEpLen,
                self.traceLen
            )
        else:
            self.memory = mem.Memory(self.memSize, self.preTrainLen)
        self.memory.pre_populate(
            self.env, self.stackSize, self.crop, self.shrink
        )
        # Build the network
        self.qNet = nw.DQN(
            hyperparams["architecture"],
            self.input_shape,
            self.env.action_space.n,
            self.learningRate,
        )
        # If applicable, build the second network for use with
        # fixed-Q and double dqn
        if self.enableFixedQ:
            self.targetQNet = nw.DQN(
                hyperparams["architecture"],
                self.input_shape,
                self.env.action_space.n,
                self.learningRate,
            )

    #-----
    # train
    #-----
    def initialize_training(self, restart=True):
        """
        Sets up the training loop.

        Parameters:
        ------------
            restart : bool
                If true, start training from the beginning. If false,
                load a saved model and continue where training last left
                off.

        Raises:
        -------
            pass

        Returns:
        --------
            None
        """
        # Start from scratch
        if restart:
            # Copy the weights from qNet to targetQNet, if applicable
            if self.enableFixedQ:
                self.targetQNet.model.set_weights(self.qNet.model.get_weights())
            # Initialize counters
            train_params = (
                0,
                0,
                self.totalRewards,
                self.memory.buffer,
                0
            )
        # Continue where we left off
        else:
            self.qNet.model = tf.keras.models.load_model(
                os.path.join(self.saveFilePath, self.ckptFile + ".h5"),
            )
            if self.enableFixedQ:
                self.targetQNet.model = tf.keras.models.load_model(
                    os.path.join(
                        self.saveFilePath, self.ckptFile + "-target.h5"
                    ),
                )
            train_params = io.load_train_params(
                self.saveFilePath, self.memory.max_size
            )
        return train_params

    #-----
    # train
    #-----
    def train(self, restart=True):
        """
        Trains the agent to play the game.

        Parameters:
        ------------
            restart : bool
                If true, start training from the beginning. If false,
                load a saved model and continue where training last left
                off.

        Raises:
        -------
            pass

        Returns:
        --------
            None
        """
        # Initialize the training loop
        abort = False
        start_ep, \
        decay_step, \
        self.totalRewards, \
        self.memory.buffer, \
        fixed_Q_step = self.initialize_training(restart)
        # Loop over desired number of training episodes
        for episode in range(start_ep, self.nEpisodes):
            print("Episode: %d / %d" % (episode + 1, self.nEpisodes))
            # Reset time spent on current episode
            step = 0
            # Track the rewards for the episode
            episode_rewards = []
            # Reset the environment
            state = self.env.reset()
            # Stack and process initial state
            state, frame_stack = frames.stack_frames(
                None,
                state,
                True,
                self.stackSize,
                self.crop,
                self.shrink
            )
            # Loop over the max amount of time the agent gets per
            # episode
            while step < self.maxEpSteps:
                print("Step: %d / %d" % (step, self.maxEpSteps), end="\r")
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
                next_state, frame_stack = frames.stack_frames(
                    frame_stack,
                    next_state,
                    False,
                    self.stackSize,
                    self.crop,
                    self.shrink,
                )
                # Save experience
                experience = (state, action, reward, next_state, done)
                self.memory.add(experience)
                # Learn from the experience
                loss = self.learn()
                # Update the targetQNet if applicable
                if self.enableFixedQ:
                    if fixed_Q_step > self.fixedQSteps:
                        fixed_Q_step = 0
                        self.targetQNet.model.set_weights(
                            self.qNet.model.get_weights()
                        )
                # Set up for next episode if we're in a terminal state
                if done:
                    # Get total reward for episode
                    tot_reward = np.sum(episode_rewards)
                    break
                # Otherwise, set up for the next step
                else:
                    state = next_state
            # Save total episode reward
            self.totalRewards.append(tot_reward)
            # Print info to screen
            print(
                "Episode: {}\n".format(episode),
                "Total Reward for episode: {}\n".format(tot_reward),
                "Training loss: {:.4f}".format(loss),
            )
            # Save the model, if applicable
            if episode % self.savePeriod == 0:
                io.model_save(
                    self.saveFilePath,
                    self.ckptFile,
                    self.qNet,
                    self.targetQNet,
                    train_params,
                    False # Whether to save the train_params or not
                )
        # Save the final, trained model now that training is done
        io.model_save(
            self.saveFilePath,
            self.ckptFile,
            self.qNet,
            self.targetQNet,
            train_params,
            True # Whether to save the train params or not
        )
