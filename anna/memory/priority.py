"""
Title:   priority.py
Author:  Jared Coughlin
Date:    8/27/19
Purpose: Contains the PriorityMemory class
Notes:
"""


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
