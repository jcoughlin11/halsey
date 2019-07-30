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
