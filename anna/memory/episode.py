"""
Title:   episode.py
Author:  Jared Coughlin
Date:    8/27/19
Purpose: Contains the EpisodeMemory class
Notes:
"""


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
    def __init__(self, maxSize, pretrainLen, preTrainNEp, traceLen):
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
        super().__init__(maxSize, pretrainLen, traceLen)
        self.preTrainNEp = self.preTrainNEp

    # -----
    # Pre-Populate
    # -----
    def pre_populate(self, env, stackSize, crop, shrink):
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

            stackSize : int
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
                None, state, True, stackSize, crop, shrink, self.traceLen)
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
