"""
Title: memory.py
Author: Jared Coughlin
Date: 7/30/19
Purpose: Contains the various memory buffer classes
Notes:
"""
import collections
import random

import numpy as np

import frames
import nnutils as nu


# ============================================
#               Memory Class
# ============================================
class Memory:
    """
    Holds and manages the experience buffer for DQNs.

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
    def __init__(self, max_size, pretrain_len, arch, traceLen):
        """
        Parameters:
        -----------
            max_size : int
                The max number of experience tuples the buffer can hold
                before it "forgets".

            pretrain_len : int
                The number of initial, sample/dummy experiences to fill
                the buffer with so we don't run into the empty memory
                problem when trying to train initially.

        Raises:
        -------
            pass

        Returns:
        --------
            pass
        """
        self.max_size = max_size
        self.pretrain_len = pretrain_len
        self.buffer = collections.deque(maxlen=self.max_size)
        self.arch = arch
        self.traceLen = traceLen

    # -----
    # Pre-Populate
    # -----
    def pre_populate(self, env, stack_size, crop, shrink):
        """
        This function initially fills the experience buffer with sample
        experience tuples to avoid the empty memory problem.

        Parameters:
        -----------
            env : gym environment
                The environment for the game that's being learned by
                the agent.

            stack_size : int
                The number of frames to stack together for temporal
                differencing.

            crop : tuple
                (top, bot, left, right) to chop off each edge of the
                frame.

            shrink : tuple
                (x, y) size of the shrunk frame.

        Raises:
        -------
            pass

        Returns:
        --------
            None
        """
        # Get initial state
        state = env.reset()
        # Process and stack initial frames
        state, frame_stack = frames.stack_frames(
            None, state, True, stack_size, crop, shrink, self.arch, self.traceLen
        )
        # Loop over the desired number of sample experiences
        for i in range(self.pretrain_len):
            # Choose a random action. randint chooses in [a,b)
            action = np.random.randint(0, env.action_space.n)
            # Take action
            next_state, reward, done, _ = env.step(action)
            # Add next state to stack of frames
            next_state, frame_stack = frames.stack_frames(
                frame_stack, next_state, False, stack_size, crop, shrink, self.arch, self.traceLen
            )
            # Add experience to memory
            self.add((state, action, reward, next_state, done))
            # If we're in a terminal state, we need to reset things
            if done:
                state = env.reset()
                state, frame_stack = frames.stack_frames(
                    None, state, True, stack_size, crop, shrink, self.arch, self.traceLen
                )
            # Otherwise, update the state and continue
            else:
                state = next_state

    # -----
    # Add
    # -----
    def add(self, experience):
        """
        Adds the newest experience tuple to the buffer.

        Parameters:
        -----------
            experience : tuple (or list of tuples in the case of an
                         RNN)
                Contains the state, action, reward, next_state, and done
                flag.

        Raises:
        -------
            pass

        Returns:
        --------
            None 
        """
        self.buffer.append(experience)

    # -----
    # Sample
    # -----
    def sample(self, batch_size):
        """
        This function returns a randomly selected subsample of size
        batch_size from the buffer. This subsample is used to train the
        DQN. Note that a deque's size is determined only from the
        elements that are in it, not from maxlen. That is, if you have a
        deque with maxlen = 10, but only one element has been added to
        it, then it's size is actually 1.

        Parameters:
        -----------
            batch_size : int
                The size of the sample to be returned.

        Raises:
        -------
            pass

        Returns:
        --------
            sample : list
                A list of randomly chosen experience tuples from the
                buffer. Chosen without replacement. The length of the
                list is batch_size.
        """
        # Choose random indices from the buffer. Make sure the
        # batch_size isn't larger than the current buffer size or np
        # will complain
        try:
            indices = np.random.choice(
                np.arange(len(self.buffer)), size=batch_size, replace=False
            )
        except ValueError:
            raise (
                "Error, need batch_size < buf_size when sampling from memory!"
            )
        return [self.buffer[i] for i in indices]


# ============================================
#           PriorityMemory Class
# ============================================
class PriorityMemory(Memory):
    """
    This class serves as the memory buffer in the case that prioritized
    experience replay is being used. It functions in much the same way
    as Memory(), but employs a SumTree() instead of a deque. This means
    that the adding and sampling methods are different.

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
    def __init__(self, max_size, pretrain_len, perParams, arch, traceLen):
        """
        Parameters:
        -----------
            max_size : int
                The max number of experience tuples the buffer can hold
                before it begins to "forget" experiences (i.e., they
                are overwritten).

            pretrain_len : int
                The number of initial, samply/dummy experiences to fill
                the buffer with so we don't run into the empty memory
                problem when trying to train initially.

            env : gym environment
                The environment for the game. Used to pre-populate the
                memory buffer.

            stack_size : int
                Number of frames to stack.

            crop : tuple
                (top, bot, left, right) to chop off each edge of the
                frame.

            shrink : tuple
                (x,y) size of the shrunk frame.

            perParams : list
                A list of alpha, beta, epsilon, and the annealment
                rate.

        Raises:
        -------
            pass

        Returns:
        --------
            pass
        """
        # Call parent's constructor
        super().__init__(max_size, pretrain_len, arch, traceLen)
        # Set per parameters
        self.perA = perParams[0]
        self.perB = perParams[1]
        self.perBAnneal = perParams[2]
        self.perE = perParams[3]
        # Overload the buffer
        self.buffer = nu.SumTree(self.max_size)
        self.upperPriority = 1.0

    # -----
    # Add
    # -----
    def add(self, experience):
        """
        This function stores the newest experience tuple, along with a
        priority, to the buffer. According to Schaul16 algorithm 1, the
        new experiences are added with a priority equal to the current
        max priority in the tree.

        Parameters:
        -----------
            experience : tuple
                Contains the state, action ,reward, next_state, and done
                flag.

        Raises:
        -------
            pass

        Returns:
        --------
            None
        """
        # Get the current max priority in the tree. Recall that the left
        # nodes hold the priority and that they are stored as the last
        # max_size elements in the array that holds the tree
        maxPriority = np.max(self.buffer.tree[-self.buffer.nLeafs :])
        # If the maxPriority is 0, then we need to set it to the
        # predefined upperPriority because a priority of 0 means that
        # the experience will never be chosen; and we want every
        # experience to have a chance at being chosen
        if maxPriority == 0:
            maxPriority = self.upperPriority
        self.buffer.add(experience, maxPriority)

    # -----
    # Sample
    # -----
    def sample(self, batchSize):
        """
        This function returns a subsample of experiences from the memory
        buffer to be used in training. The probability for a particular
        experience to be chosen is given by equation 1 in Schaul16. The
        details of how to sample from the sumtree are given in Appendix
        B.2.1: Proportional prioritization in Schaul16. Essentially, we
        break up the range [0, priority_total] into batchSize segments
        of equal size. We then uniformly choose a value from each
        segment and get the experiences that correspond to each of these
        sampled values.

        Parameters:
        -----------
            batchSize : int
                The size of the sample to return.

        Raises:
        -------
            pass

        Returns:
        --------
            indices : ndarray
                An array of tree indices corresponding to the sampled
                experiences.

            experiences : list
                A list of batchSize experiences chosen with
                probabilities given by Schaul16 equation 1.

            isWeights : ndarray
                An array containing the IS weights for each sampled
                experience.
        """
        # We need to return the selected samples (to be used in
        # training), the indices of these samples (so that the tree can
        # be properly updated), and the importance sampling weights to
        # be used in training
        indices = np.zeros((batchSize,), dtype=np.int)
        priorities = np.zeros((batchSize, 1))
        experiences = []
        # We need to break up the range [0, p_tot] equally into
        # batchSize segments, so here we get the width of each segment
        segmentWidth = self.buffer.total_priority / batchSize
        # Anneal the strength of the IS weights (cap the parameter at 1)
        self.perB = np.min([1.0, self.perB + self.perBAnneal])
        # Loop over the desired number of samples
        for i in range(batchSize):
            # We need to uniformly select a value from each segment, so
            # here we get the lower and upper bounds of the segment
            lowerBound = i * segmentWidth
            upperBound = (i + 1) * segmentWidth
            # Choose a value from within the segment
            value = np.random.uniform(lowerBound, upperBound)
            # Retrieve the experience whose priority matches value from
            # the tree
            index, priority, experience = self.buffer.get_leaf(value)
            indices[i] = index
            priorities[i, 0] = priority
            experiences.append(experience)
        # Calculate the importance sampling weights
        samplingProbabilities = priorities / self.buffer.total_priority
        isWeights = np.power(batchSize * samplingProbabilities, -self.perB)
        isWeights = isWeights / np.max(isWeights)
        return indices, experiences, isWeights

    # -----
    # Update
    # -----
    def update(self, indices, absErrors):
        """
        This function uses the new errors generated from training in
        order to update the priorities for those experiences that were
        selected in sample().

        Parameters:
        -----------
            indices : ndarray
                Array of tree indices corresponding to those experiences
                used in the training batch.

            absErrors : ndarray
                Array of the absolute value of the TD errors for the
                chosen experiences.

        Raises:
        -------
            pass

        Returns:
        --------
            None
        """
        # Calculate priorities from errors (proportional prioritization)
        priorities = absErrors + self.perE
        # Clip the errors for stability
        priorities = np.minimum(priorities, self.upperPriority)
        # Apply alpha
        priorities = np.power(priorities, self.perA)
        # Update the tree
        for ind, p in zip(indices, priorities):
            self.buffer.update(ind, p)


# ============================================
#                EpisodeMemory
# ============================================
class EpisodeMemory(Memory):
    """
    Same idea as the Memory class, but holds entire episodes in its
    buffer instead of individual experience tuples. As such, episodes
    are sampled rather than random experience tuples. A random string,
    or trace, of frames of a particular length is chosen from each
    episode. These traces are used to train a RNN.

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
    def __init__(self, max_size, pretrain_len, preTrainNEp, traceLen, arch):
        """
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
        # Call parent's constructor
        super().__init__(max_size, pretrain_len, arch, traceLen)
        self.preTrainNEp = self.preTrainNEp

    # -----
    # Pre-Populate
    # -----
    def pre_populate(self, env, stack_size, crop, shrink):
        """
        This function initially fills the experience buffer with sample
        episodes in order to avoid the empty memory problem. It's a
        little frustrating since it's so similar to Memory's
        pre_populate, but I figured this was better than doing an
        instance check or passing a flag to that function.

        Parameters:
        -----------
            env : gym environment
                The environment for the game that's being learned by
                the agent.

            stack_size : int
                The number of frames to stack together for temporal
                differencing.

            crop : tuple
                (top, bot, left, right) to chop off each edge of the
                frame.

            shrink : tuple
                (x, y) size of the shrunk frame.

        Raises:
        -------
            pass

        Returns:
        --------
            None
        """
        # Loop over the desired number of sample episodes
        for i in range(self.preTrainNEp):
            # Get initial state
            state = env.reset()
            # Process and stack initial frames
            state, frame_stack = frames.stack_frames(
                None, state, True, stack_size, crop, shrink, self.arch, self.traceLen
            )
            # Clear episode buffer
            episodeBuffer = []
            # Loop over the max number of steps we can take per episode
            for j in range(self.pretrain_len):
                # Choose a random action
                action = env.action_space.sample()
                # Take action
                next_state, reward, done, _ = env.step(action)
                # Add next state to stack of frames
                next_state, frame_stack = frames.stack_frames(
                    frame_stack, next_state, False, stack_size, crop, shrink, self.arch, self.traceLen
                )
                # Add experience to episode buffer
                episodeBuffer.append((state, action, reward, next_state, done))
                # If we're in a terminal state, go to next episode
                if done:
                    break
                # Otherwise, update the state and continue
                else:
                    state = next_state
            # When done with the current episode, add the experience to
            # buffer
            self.add(episodeBuffer)

    # -----
    # sample
    # -----
    def sample(self, batchSize):
        """
        Randomly selects episodes from the memory buffer and randomly
        chooses traces of the desired length from each episode in order
        to train on them.

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
        # Set up list for holding traces
        traces = []
        # Choose random episodes
        chosenEpisodes = random.sample(self.buffer, batchSize)
        # Select random traces of the desired length from each of the
        # chosen episodes
        for ep in chosenEpisodes:
            # Case 1: episode is long enough for desired trace
            if len(ep) >= self.traceLen:
                # Get starting index. The upper limit is exclusive,
                # hence the +1
                ind = np.random.randint(0, len(ep) + 1 - self.traceLen)
                # Get trace
                traces.append(ep[ind : ind + self.traceLen])
            # Case 2: it isn't
            else:
                # Extract as many experiences as we can
                partialTrace = chosenEpisodes[i][ind:]
                # NOTE: I'm not sure this is the right way to handle
                # this! I'm repeating the last experience until we fill
                # the trace, but this breaks the sequential nature of
                # the data that an RNN needs. Can I have a dynamic
                # batch size? Or should I just choose another episode?
                # What if there are no episodes of the right length?
                while len(partialTrace) < self.traceLen:
                    partialTrace.append(chosenEpisodes[i][-1])
                traces.append(partialTrace)
        return traces
