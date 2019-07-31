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
    def initialize_training(self, restart):
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
            # Reset the environment
            state = self.env.reset()
            # Stack and process initial state
            state, frameStack = frames.stack_frames(
                None,
                state,
                True,
                self.stackSize,
                self.crop,
                self.shrink
            )
            # Initialize parameters: startEp, decayStep, step,
            # fixedQStep, totalRewards, epRewards, state, frameStack,
            # and buffer
            trainParams = (
                0,
                0,
                0,
                0,
                [],
                [],
                state,
                frameStack,
                self.memory.buffer
            )
        # Continue where we left off
        else:
            # Load network
            self.qNet.model = tf.keras.models.load_model(
                os.path.join(self.saveFilePath, self.ckptFile + ".h5"),
            )
            # Load target network, if applicable
            if self.enableFixedQ:
                self.targetQNet.model = tf.keras.models.load_model(
                    os.path.join(
                        self.saveFilePath, self.ckptFile + "-target.h5"
                    ),
                )
            # Load training parameters
            trainParams = io.load_train_params(
                self.saveFilePath,
                self.memory.max_len
            )
        return trainParams

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
        earlyStop = False
        trainParams = None
        startEp, \
        decayStep, \
        step, \
        fixedQStep,
        self.totalRewards, \
        episodeRewards, \
        state, \
        frameStack, \
        self.memory.buffer = self.initialize_training(restart)
        # Loop over desired number of training episodes
        for episode in range(startEp, self.nEpisodes):
            print("Episode: %d / %d" % (episode + 1, self.nEpisodes))
            if episode > startEp:
                # Reset time spent on current episode
                step = 0
                # Track the rewards for the episode
                episodeRewards = []
                # Reset the environment
                state = self.env.reset()
                # Stack and process initial state
                state, frameStack = frames.stack_frames(
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
                # Check for early stop
                if nu.check_early_stop():
                    earlyStop = True
                    trainParams = (
                        episode,
                        decayStep,
                        step,
                        fixedQStep,
                        self.totalRewards,
                        episodeRewards,
                        state,
                        frameStack
                        self.memory.buffer
                    )
                    break
                print("Step: %d / %d" % (step, self.maxEpSteps), end="\r")
                # Increase step counters
                step += 1
                decayStep += 1
                fixedQStep += 1
                # Choose an action
                action = self.choose_action(state, decayStep)
                # Perform action
                nextState, reward, done, _ = self.env.step(action)
                # Track the reward
                episodeRewards.append(reward)
                # Add the next state to the stack of frames
                nextState, frameStack = frames.stack_frames(
                    frameStack,
                    nextState,
                    False,
                    self.stackSize,
                    self.crop,
                    self.shrink,
                )
                # Save experience
                experience = (state, action, reward, nextState, done)
                self.memory.add(experience)
                # Learn from the experience
                loss = self.learn()
                # Update the targetQNet if applicable
                if self.enableFixedQ:
                    if fixedQStep > self.fixedQSteps:
                        fixedQStep = 0
                        self.targetQNet.model.set_weights(
                            self.qNet.model.get_weights()
                        )
                # Set up for next episode if we're in a terminal state
                if done:
                    # Get total reward for episode
                    totReward = np.sum(episodeRewards)
                    break
                # Otherwise, set up for the next step
                else:
                    state = nextState
            # Save total episode reward
            self.totalRewards.append(totReward)
            # Print info to screen
            if not earlyStop:
                print(
                    "Episode: {}\n".format(episode),
                    "Total Reward for episode: {}\n".format(totReward),
                    "Training loss: {:.4f}".format(loss),
                )
            # Save the model, if applicable
            if episode % self.savePeriod == 0 or earlyStop:
                io.save_model(
                    self.saveFilePath,
                    self.ckptFile,
                    self.qNet,
                    self.targetQNet,
                    trainParams,
                    True
                )
        # Save the final, trained model now that training is done
        io.save_model(
            self.saveFilePath,
            self.ckptFile,
            self.qNet,
            self.targetQNet,
            trainParams,
            False
        )
