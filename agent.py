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
        self.inputShape = (
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
            self.inputShape,
            self.env.action_space.n,
            self.learningRate,
        )
        # If applicable, build the second network for use with
        # fixed-Q and double dqn
        if self.enableFixedQ:
            self.targetQNet = nw.DQN(
                hyperparams["architecture"],
                self.inputShape,
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
                self.memory.max_size
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
        fixedQStep, \
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
            # Print info to screen
            if not earlyStop:
                # Save total episode reward
                self.totalRewards.append(totReward)
                print(
                    "Episode: {}\n".format(episode),
                    "Total Reward for episode: {}\n".format(totReward),
                    "Training loss: {:.4f}".format(loss),
                )
            # Save the model, if applicable
            if episode % self.savePeriod == 0 or earlyStop:
                trainParams = (
                    episode,
                    decayStep,
                    step,
                    fixedQStep,
                    self.totalRewards,
                    episodeRewards,
                    state,
                    frameStack,
                    self.memory.buffer
                )
                io.save_model(
                    self.saveFilePath,
                    self.ckptFile,
                    self.qNet,
                    self.targetQNet,
                    trainParams,
                    True
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
            False
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
            # Keras expects a group of samples of the specified shape,
            # even if there's just one sample, so we need to reshape
            state = state.reshape(
                (
                    1,
                    state.shape[0],
                    state.shape[1],
                    state.shape[2]
                )
            )
            # Get the beliefs in each action for the current state
            Q_vals = self.qNet.model.predict_on_batch(state)
            # Choose the one with the highest Q value
            action = np.argmax(Q_vals)
        return action

    #-----
    # learn
    #-----
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
        states = np.zeros([self.batchSize] + [d for d in self.inputShape])
        actions = np.zeros(self.batchSize, dtype=np.int)
        rewards = np.zeros(self.batchSize)
        nextStates = np.zeros([self.batchSize] + [d for d in self.inputShape])
        dones = np.zeros(self.batchSize, dtype=np.bool)
        # Get batch of experiences
        if self.enablePer:
            treeInds, sample, isWeights = self.memory.sample(self.batchSize)
        else:
            sample = self.memory.sample(self.batchSize)
        # Unpack the batch
        for i, s in enumerate(sample):
            states[i] = s[0]
            actions[i] = s[1]
            rewards[i] = s[2]
            nextStates[i] = s[3]
            dones[i] = s[4]
        # Update the weights based on which technique is being used
        # Double DQN
        if self.enableDoubleDQN:
            loss = self.double_dqn_learn(
                states,
                actions,
                rewards,
                nextStates,
                dones
            )
        # Fixed-Q
        elif self.enableFixedQ:
            loss = self.fixed_q_learn(
                states,
                actions,
                rewards,
                nextStates,
                dones
            )
        # Standard dqn
        else:
            loss = self.dqn_learn(
                states,
                actions,
                rewards,
                nextStates,
                dones
            )
        return loss

    #-----
    # dqn_learn
    #-----
    def dqn_learn(self, states, actions, rewards, nextStates, dones):
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
        # Get qNext: estimate of best trajectory obtained by playing
        # optimally from the next state. This is used in the estimate
        # of Q-target
        # Standard dqn
        qNext = self.qNet.model.predict_on_batch(nextStates)
        # Get Q-target: estimate of Q-values obtained from sample
        # trajectories. Vectorized using the dones array as a mask
        qTarget = self.qNet.model.predict_on_batch(states)
        # Update only the entry for the current state-action pair in the
        # vector of Q-values that corresponds to the chosen action. For
        # a terminal state it's just the reward, and otherwise we use
        # the Bellmann equation
        doneInds = np.where(dones)
        nDoneInds = np.where(~dones)
        qTarget[doneInds, actions[doneInds]] = rewards[doneInds]
        qTarget[nDoneInds, actions[nDoneInds]] = rewards[nDoneInds] + \
            self.discountRate * np.amax(qNext[nDoneInds])
        # Update the network weights
        loss = self.qNet.model.train_on_batch(states, qTarget)
        return loss

    #-----
    # double_dqn_learn
    #-----
    def double_dqn_learn(self, states, actions, rewards, nextStates, dones):
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
        qTarget = self.qNet.model.predict_on_batch(states)
        doneInds = np.where(dones)
        nDoneInds = np.where(~dones)
        qTarget[doneInds, actions[doneInds]] = rewards[doneInds]
        qTarget[nDoneInds, actions[nDoneInds]] = rewards[nDoneInds] + \
            self.discountRate * qNext[nDoneInds, nextActions]
        # Update the network weights
        loss = self.qNet.model.train_on_batch(states, qTarget)
        return loss

    #-----
    # fixed_q_learn
    #-----
    def fixed_q_learn(self, states, actions, rewards, nextStates, dones):
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
        qTarget = self.qNet.model.predict_on_batch(states)
        doneInds = np.where(dones)
        nDoneInds = np.where(~dones)
        qTarget[doneInds, actions[doneInds]] = rewards[doneInds]
        qTarget[nDoneInds, actions[nDoneInds]] = rewards[nDoneInds] + \
            self.discountRate * np.amax(qNext[nDoneInds])
        # Update the network weights
        loss = self.qNet.model.train_on_batch(states, qTarget)
        return loss
