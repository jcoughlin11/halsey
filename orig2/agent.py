"""
Title:   agent.py
Author:  Jared Coughlin
Date:    1/24/19
Purpose: Contains the Agent class, which is the object that learns
Notes:
"""
import os
import sys
import time

import numpy as np
import tensorflow as tf

import frames
import memory as mem
import nnetworks as nw
import nnio as io
import nnutils as nu


# ============================================
#                    Agent
# ============================================
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
        self.arch = hyperparams['architecture']
        self.batchSize = hyperparams["batch_size"]
        self.ckptFile = hyperparams["ckpt_file"]
        self.cropBot = hyperparams["crop_bot"]
        self.cropLeft = hyperparams["crop_left"]
        self.cropRight = hyperparams["crop_right"]
        self.cropTop = hyperparams["crop_top"]
        self.discountRate = hyperparams["discount"]
        self.enableDoubleDQN = hyperparams["enable_double_dqn"]
        self.enableFixedQ = hyperparams["enable_fixed_Q"]
        self.enablePer = hyperparams["enable_per"]
        self.env = env
        self.epsDecayRate = hyperparams["eps_decay_rate"]
        self.epsilonStart = hyperparams["epsilon_start"]
        self.epsilonStop = hyperparams["epsilon_stop"]
        self.fixedQSteps = hyperparams["fixed_Q_steps"]
        self.learningRate = hyperparams["learning_rate"]
        self.loss = hyperparams["loss"]
        self.maxEpSteps = hyperparams["max_episode_steps"]
        self.memSize = hyperparams["memory_size"]
        self.nEpisodes = hyperparams["n_episodes"]
        self.optimizer = hyperparams["optimizer"]
        self.perA = hyperparams["per_a"]
        self.perB = hyperparams["per_b"]
        self.perBAnneal = hyperparams["per_b_anneal"]
        self.perE = hyperparams["per_e"]
        self.preTrainNEp = hyperparams["pretrain_n_eps"]
        self.preTrainLen = hyperparams["pretrain_len"]
        self.qNet = None
        self.renderFlag = hyperparams["render_flag"]
        self.restartTraining = hyperparams["restart_training"]
        self.savePeriod = hyperparams["save_period"]
        self.saveFilePath = hyperparams["save_path"]
        self.shrinkCols = hyperparams["shrink_cols"]
        self.shrinkRows = hyperparams["shrink_rows"]
        self.stackSize = hyperparams["n_stacked_frames"]
        self.targetQNet = None
        self.timeLimit = hyperparams["time_limit"]
        self.totalRewards = []
        self.traceLen = hyperparams["trace_len"]
        # Seed the rng
        np.random.seed(int(time.time()))
        # Set up tuples for preprocessed frame sizes
        self.crop = (self.cropTop, self.cropBot, self.cropLeft, self.cropRight)
        self.shrink = (self.shrinkRows, self.shrinkCols)
        # Set the size of the input frame stack
        # See nnetworks.py, build_rnn1_net
        if self.arch == 'rnn1':
            self.inputShape = (
                self.traceLen,
                self.shrinkRows,
                self.shrinkCols,
            )
        else:
            self.inputShape = (self.shrinkRows, self.shrinkCols, self.stackSize)
        # Set up memory
        if self.enablePer:
            perParams = [self.perA, self.perB, self.perBAnneal, self.perE]
            self.memory = mem.PriorityMemory(
                self.memSize, self.preTrainLen, perParams, self.arch, self.traceLen
            )
        elif self.arch == "rnn1":
            self.memory = mem.EpisodeMemory(
                self.memSize,
                self.preTrainLen,
                self.preTrainNEp,
                self.traceLen,
                self.arch
            )
        else:
            self.memory = mem.Memory(self.memSize, self.preTrainLen, self.arch, self.traceLen)
        self.memory.pre_populate(
            self.env, self.stackSize, self.crop, self.shrink
        )
        # Build the network
        self.qNet = nw.DQN(
            hyperparams["architecture"],
            self.inputShape,
            self.env.action_space.n,
            self.learningRate,
            self.optimizer,
            self.loss,
        )
        # If applicable, build the second network for use with
        # fixed-Q and double dqn
        if self.enableFixedQ:
            self.targetQNet = nw.DQN(
                hyperparams["architecture"],
                self.inputShape,
                self.env.action_space.n,
                self.learningRate,
                self.optimizer,
                self.loss,
            )

    # -----
    # train
    # -----
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
            # state.shape=(shrinkRows, shrinkCols, stackSize) for 
            # non-RNN and state = trace =>
            # trace.shape=(traceLen, nrows, ncols) for RNN
            state, frameStack = frames.stack_frames(
                None, state, True, self.stackSize, self.crop, self.shrink,
                self.arch, self.traceLen
            )
            # Initialize parameters: startEp, decayStep, step,
            # fixedQStep, totalRewards, epRewards, state, frameStack,
            # and buffer, and epBuffer
            if self.arch == 'rnn1':
                epBuffer = []
            else:
                epBuffer = None
            trainParams = (
                0,
                0,
                0,
                0,
                [],
                [],
                state,
                frameStack,
                self.memory.buffer,
                epBuffer
            )
        # Continue where we left off
        else:
            # Load network
            self.qNet.model = tf.keras.models.load_model(
                os.path.join(self.saveFilePath, self.ckptFile + ".h5")
            )
            # Load target network, if applicable
            if self.enableFixedQ:
                self.targetQNet.model = tf.keras.models.load_model(
                    os.path.join(
                        self.saveFilePath, self.ckptFile + "-target.h5"
                    )
                )
            # Load training parameters
            trainParams = io.load_train_params(
                self.saveFilePath, self.memory.max_size
            )
        return trainParams

    # -----
    # train
    # -----
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
        startTime = time.time()
        earlyStop = False
        trainParams = None
        startEp, \
        decayStep, \
        step, \
        fixedQStep, \
        self.totalRewards, \
        episodeRewards, \
        state, \
        frameStack, \
        self.memory.buffer, \
        epBuffer = self.initialize_training(restart)
        # Loop over desired number of training episodes
        for episode in range(startEp, self.nEpisodes):
            print("Episode: %d / %d" % (episode + 1, self.nEpisodes))
            if episode > startEp:
                # Clear episode buffer for rnns
                if self.arch == 'rnn1':
                    epBuffer = []
                # Reset time spent on current episode
                step = 0
                # Track the rewards for the episode
                episodeRewards = []
                # Reset the environment
                state = self.env.reset()
                # Stack and process initial state
                # state.shape=(shrinkRows, shrinkCols, stackSize)
                state, frameStack = frames.stack_frames(
                    None, state, True, self.stackSize, self.crop, self.shrink,
                    self.arch, self.traceLen
                )
            # Loop over the max amount of time the agent gets per
            # episode
            while step < self.maxEpSteps:
                # Check for early stop
                if nu.check_early_stop(self.saveFilePath):
                    earlyStop = True
                    break
                # Check for time limit reached
                if nu.time_limit_reached(startTime, self.timeLimit):
                    earlyStop = True
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
                # state.shape=(shrinkRows, shrinkCols, stackSize)
                nextState, frameStack = frames.stack_frames(
                    frameStack,
                    nextState,
                    False,
                    self.stackSize,
                    self.crop,
                    self.shrink,
                    self.arch,
                    self.traceLen
                )
                # Save experience
                if self.arch == 'rnn1':
                    # For an RNN, I have stack_frames returning a trace
                    # of shape (traceLen, rows, cols), but the memory
                    # buffer needs to hold episodes, where each episode
                    # is comprised of individual frames. As such, I'm
                    # taking the current addition to each slice so that
                    # the experience tuple is comprised of one frame
                    # for the states and nextStates
                    experience = (state[-1], action, reward, nextState[-1], done)
                    epBuffer.append(experience)
                else:
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
            # Save episode in the case of a time-out for an rnn
            if self.arch == 'rnn1' and not earlyStop:
                self.memory.add(epBuffer)
            # Print info to screen
            if not earlyStop:
                # Save total episode reward
                self.totalRewards.append(totReward)
                print(
                    "Episode: {}\n".format(episode),
                    "Total Reward for episode: {}\n".format(totReward),
                    "Training loss: {:.4f}".format(loss),
                )
            # Package the training params
            trainParams = (
                episode,
                decayStep,
                step,
                fixedQStep,
                self.totalRewards,
                episodeRewards,
                state,
                frameStack,
                self.memory.buffer,
                epBuffer
            )
            # Save the model, if applicable
            if episode % self.savePeriod == 0 or earlyStop:
                io.save_model(
                    self.saveFilePath,
                    self.ckptFile,
                    self.qNet,
                    self.targetQNet,
                    trainParams,
                    True,
                )
                if earlyStop:
                    sys.exit()
        # Save the final, trained model now that training is done
        io.save_model(
            self.saveFilePath,
            self.ckptFile,
            self.qNet,
            self.targetQNet,
            trainParams,
            False,
        )

    # -----
    # Choose Action
    # -----
    def choose_action(self, state, decayStep):
        """
        Uses the current state and the agent's current knowledge in
        order to choose an action. It employs the epsilon greedy
        strategy to handle exploration vs. exploitation.

        Parameters:
        ------------
            state : ndarray
                The tensor of stacked frames.

            decayStep : int
                The overall training step. This is used to cause the
                agent to favor exploitation in the long-run.

        Raises:
        -------
            pass

        Returns:
        --------
            action : int
                The agent's choice of what to do based on the current
                state.
        """
        # Choose a random number from uniform distribution between 0 and
        # 1. This is the probability that we exploit the knowledge we
        # already have
        exploitProb = np.random.random()
        # Get the explore probability. This probability decays over time
        # (but stops at eps_stop so we always have some chance of trying
        # something new) as the agent learns
        exploreProb = self.epsilonStop + (
            self.epsilonStart - self.epsilonStop
        ) * np.exp(-self.epsDecayRate * decayStep)
        # Explore
        if exploreProb >= exploitProb:
            # Choose randomly
            action = self.env.action_space.sample()
        # Exploit
        else:
            # Keras requires the batch size as a part of the input
            # shape when training or predicting, but not setting up
            # the network
            state = state.reshape([1] + list(state.shape[:]))
            # Get the beliefs in each action for the current state
            Q_vals = self.qNet.model.predict_on_batch(state)
            # Choose the one with the highest Q value
            action = np.argmax(Q_vals)
        return action

    # -----
    # learn
    # -----
    def learn(self):
        """
        Samples from the experience buffer and calls the correct learn
        method in order to update the network's weights.

        Parameters:
        -----------
            pass

        Raises:
        -------
            pass

        Returns:
        --------
            pass
        """
        # Set up the unpack arrays
        if self.arch != 'rnn1':
            states = np.zeros([self.batchSize] + list(self.inputShape[:]))
            actions = np.zeros(self.batchSize, dtype=np.int)
            rewards = np.zeros(self.batchSize)
            nextStates = np.zeros([self.batchSize] + list(self.inputShape[:]))
            dones = np.zeros(self.batchSize, dtype=np.bool)
        else:
            states = np.zeros([self.batchSize] + list(self.inputShape[:]))
            actions = np.zeros((self.batchSize, self.traceLen), dtype=np.int)
            rewards = np.zeros((self.batchSize, self.traceLen))
            nextStates = np.zeros([self.batchSize] + list(self.inputShape[:]))
            dones = np.zeros((self.batchSize, self.traceLen), dtype=np.bool)
        # Get batch of experiences
        if self.enablePer:
            treeInds, sample, isWeights = self.memory.sample(self.batchSize)
        else:
            sample = self.memory.sample(self.batchSize)
            isWeights = None
        # Unpack the batch
        if self.arch != 'rnn1':
            for i, s in enumerate(sample):
                states[i] = s[0]
                actions[i] = s[1]
                rewards[i] = s[2]
                nextStates[i] = s[3]
                dones[i] = s[4]
        else:
            for i, episode in enumerate(sample):
                for j, experience in enumerate(episode):
                    states[i][j] = experience[0]
                    actions[i][j] = experiences[1]
                    rewards[i][j] = experiences[2]
                    nextStates[i][j] = experiences[3]
                    dones[i][j] = experiences[4]
        # Update the weights based on which technique is being used
        # Double DQN
        if self.enableDoubleDQN:
            loss, absError = self.double_dqn_learn(
                states, actions, rewards, nextStates, dones, isWeights.flatten()
            )
        # Fixed-Q
        elif self.enableFixedQ:
            loss, absError = self.fixed_q_learn(
                states, actions, rewards, nextStates, dones, isWeights.flatten()
            )
        # Standard dqn
        else:
            loss, absError = self.dqn_learn(
                states, actions, rewards, nextStates, dones, isWeights
            )
        # Update the priorities
        if self.enablePer:
            self.memory.update(treeInds, absError)
        return loss

    # -----
    # dqn_learn
    # -----
    def dqn_learn(self, states, actions, rewards, nextStates, dones, isWeights):
        """
        The estimates of the max discounted future rewards (qTarget) are
        the "labels" assigned to the input states.

        Basically, the network holds the current beliefs for how well
        we can do by taking a certain action in a certain state. The
        Bellmann equation provides a way to estimate, via discounted
        future rewards obtained from the sample trajectories, how well
        we can do playing optimally from the state that the chosen
        action brings us to. If this trajectory is bad, we lower the
        Q value for the current state-action pair. If it's good, then we
        increase it.

        But we only change the entry for the current state-action pair
        because the current sample trajectory doesn't tell us anything
        about what would have happened had we chosen a different action
        for the current state, and we don't know the true Q-vectors
        ahead of time. To update those other entries, we need a
        different sample. This is why it takes so many training games to
        get a good Q-table (network).

        See Mnih13 algorithm 1 for the calculation of qTarget.

        See https://keon.io/deep-q-learning/ for implementation logic.

        Note: The way OpenAI baselines does this is better.

        Parameters:
        -----------
            pass

        Raises:
        -------
            pass

        Returns:
        --------
            pass
        """
        # Get network's current guesses for Q values
        qPred = self.qNet.model.predict_on_batch(states)
        # Get qNext: estimate of best trajectory obtained by playing
        # optimally from the next state. This is used in the estimate
        # of Q-target
        qNext = self.qNet.model.predict_on_batch(nextStates)
        # Update only the entry for the current state-action pair in the
        # vector of Q-values that corresponds to the chosen action. For
        # a terminal state it's just the reward, and otherwise we use
        # the Bellmann equation
        doneInds = np.where(dones)
        nDoneInds = np.where(~dones)
        # This third array is needed so I can get absError, otherwise
        # just the specified entries of qPred could be changed
        qTarget = np.zeros(qPred.shape)
        qTarget[doneInds, actions[doneInds]] = rewards[doneInds]
        qTarget[nDoneInds, actions[nDoneInds]] = rewards[
            nDoneInds
        ] + self.discountRate * np.amax(qNext[nDoneInds])
        # Fill in qTarget with the unaffected Q values. This is so the
        # TD error for those terms is 0, since they did not change.
        # Otherwise, the TD error for those terms would be equal to
        # the original Q value for that state-action entry
        qTarget[qTarget == 0] = qPred[qTarget == 0]
        # Get the absolute value of the TD error for use in per. The
        # sum is so we only get 1 value per sample, since the priority
        # for each experience is just a float, not a sequence
        absError = tf.reduce_sum(tf.abs(qTarget - qPred), axis=1)
        # Update the network weights
        loss = self.qNet.model.train_on_batch(
            states, qTarget, sample_weight=isWeights
        )
        return loss, absError

    # -----
    # double_dqn_learn
    # -----
    def double_dqn_learn(
        self, states, actions, rewards, nextStates, dones, isWeights
    ):
        """
        Double dqn attempts to deal with the following issue: when we
        choose the action that gives rise to the highest Q value for the
        next state, how do we know that that's actually the best action?

        Since we're learning from experience, our estimated Q values
        depend on which actions have been tried and which neighboring
        states have been visited.

        As such, double dqn separates out the estimate of the Q value
        and the determination of the best action to take at the next
        state. We use our primary network to choose an action for the
        next state and then pass that action to our target network,
        which handles calculating the target Q value.

        For non-terminal states, the target value is:
        y_i = r_{i+1} + gamma * \
            Q(s_{i+1}, argmax_a(Q(s_{i+1}, a; theta_i)); theta_i')

        See van Hasselt 2015 and the dqn_learn header.

        Parameters:
        -----------
            pass

        Raises:
        -------
            pass

        Returns:
        --------
            pass

        """
        # Use primary network to generate qNext values for action
        # selection
        qNextPrimary = self.qNet.model.predict_on_batch(nextStates)
        # Get actions for next state
        nextActions = np.argmax(qNextPrimary, axis=1)
        # Use the target network and the actions chosen by the primary
        # network to get the qNext values
        qNext = self.targetQNet.model.predict_on_batch(nextStates)
        # Now get targetQ values as is done in dqn_learn
        qPred = self.qNet.model.predict_on_batch(states)
        qTarget = np.zeros(qPred.shape)
        doneInds = np.where(dones)
        nDoneInds = np.where(~dones)
        qTarget[doneInds, actions[doneInds]] = rewards[doneInds]
        qTarget[nDoneInds, actions[nDoneInds]] = (
            rewards[nDoneInds]
            + self.discountRate * qNext[nDoneInds, nextActions[nDoneInds]]
        )
        qTarget[qTarget == 0] = qPred[qTarget == 0]
        # Get abs error
        absError = tf.reduce_sum(tf.abs(qTarget - qPred), axis=1)
        # Update the network weights
        loss = self.qNet.model.train_on_batch(
            states, qTarget, sample_weight=isWeights
        )
        return loss, absError

    # -----
    # fixed_q_learn
    # -----
    def fixed_q_learn(
        self, states, actions, rewards, nextStates, dones, isWeights
    ):
        """
        In DQL the qTargets (labels) are determined from the same
        network that they are being used to update. As such, there can
        be a lot of noise due to values constantly jumping wildly. This
        affects the speed of convergence.

        In fixed-Q, a second network is used, called the target
        network. It's used to determine the qTargets (hence its name).
        These labels are then passed to the primary network so its
        weights can be updated. The weights in the primary network are
        copied over to the target network only every N steps. Due to
        this, there is far less jumping around in the target network,
        which makes its predicted labels more stable. This, in turn,
        speeds up convergence for the primary network.

        See Lillicrap et al. 2016.

        Parameters:
        -----------
            pass

        Raises:
        -------
            pass

        Returns:
        --------
            pass
        """
        # Use the target network to generate the qTargets
        qNext = self.targetQNet.model.predict_on_batch(nextStates)
        # Get the qTarget values according to dqn_learn
        qPred = self.qNet.model.predict_on_batch(states)
        qTarget = np.zeros(qPred.shape)
        doneInds = np.where(dones)
        nDoneInds = np.where(~dones)
        qTarget[doneInds, actions[doneInds]] = rewards[doneInds]
        qTarget[nDoneInds, actions[nDoneInds]] = rewards[
            nDoneInds
        ] + self.discountRate * np.amax(qNext[nDoneInds])
        qTarget[qTarget == 0] = qPred[qTarget == 0]
        # Get abs error
        absError = tf.reduce_sum(tf.abs(qTarget - qPred), axis=1)
        # Update the network weights
        loss = self.qNet.model.train_on_batch(
            states, qTarget, sample_weight=isWeights
        )
        return loss, absError
