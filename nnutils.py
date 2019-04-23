"""
Title: nnutils.py
Author: Jared Coughlin
Date: 1/24/19
Purpose: Contains utility and helper classes related to neural networks
Notes:
"""
import collections
import os
import sys
import warnings

import numpy as np
import skimage



# Skimage produces a lot of warnings
warnings.filterwarnings('ignore')

# Architecture register
archRegister = ['conv1',
                'dueling1']



#============================================
#           param_file_registers
#============================================
def param_file_registers():
    """
    This function sets up the registers containing the types associated with each
    parameter.
    """
    float_params = ['learning_rate',
                    'epsilon_start',
                    'epsilon_stop',
                    'eps_decay_rate',
                    'discount',
                    'per_a',
                    'per_b',
                    'per_e',
                    'per_b_anneal']
    int_params = ['n_episodes',
                    'max_steps',
                    'batch_size',
                    'memory_size',
                    'crop_top',
                    'crop_bot',
                    'crop_left',
                    'crop_right',
                    'nstacked_frames',
                    'train_flag',
                    'render_flag',
                    'shrink_rows',
                    'shrink_cols',
                    'pretrain_len',
                    'save_period',
                    'restart_training',
                    'test_flag',
                    'fixed_Q_steps',
                    'fixed_Q',
                    'double_dqn',
                    'per']
    string_params = ['save_path',
                     'ckpt_file',
                     'env_name',
                     'architecture']
    type_register = {'floats' : float_params,
                    'ints' : int_params,
                    'strings' : string_params}
    return type_register



#============================================
#        check_agent_option_conflicts
#============================================
def check_agent_option_conflicts(params):
    """
    This function checks for any conflicts between techniques that have been turned on
    in the parameter file.

    Parameters:
    -----------
        params : dict
            The parameters read in from the parameter file. It was easier to pass the
            whole dict rather than specific options since I don't yet know all of the
            options that will be included

    Returns:
    --------
        conflict_flag : int
            If 0, a conflict was found and the code will abort. The error message is
            generated nad printed from within this function. If 1, then everything is
            fine (hopefully!)
    """
    conflict_flag = 1
    # Double DQN requires fixed-Q
    if (params['double_dqn'] == 1) and (params['fixed_Q'] != 1):
        print("Error, double dqn requires the use of fixed Q!")
        conflict_flag = 0
    return conflict_flag



#============================================
#             read_hyperparams
#============================================
def read_hyperparams(fname):
    """
    This function reads in the parameter file that contains the network's hyperparameters.
    The layout is:
    learning_rate    : learningRate (float, network learning rate)
    n_episodes       : nEpisodes (int, number of episodes to train for)
    max_steps        : maxEpisodeSteps (int, max number of steps per episode)
    batch_size       : batchSize (int, size of batches used for training)
    epsilon_start    : epsilonStart (float, initial value of explore-exploit parameter)
    epsilon_stop     : epsilonStop (float, min value of explore-exploit parameter)
    eps_decay_rate   : epsDecayRate (float, rate at which explore-exploit param decays)
    discount         : gamma (float, reward discount rate)
    memory_size      : memSize (int, max number of experiences to store in memory buffer)
    train_flag       : trainFlag (bool, if True, train the network. If False, load saved)
    render_flag      : renderFlag (bool, if True, render scene during testing)
    crop_top         : cropTop (int, number of rows to chop off top of frame)
    crop_bot         : cropBot (int, number of rows to chop off bottom of frame)
    crop_left        : cropLeft (int, number of cols to chop off left of frame)
    crop_right       : cropRight (int, number of cols to chop off right of frame)
    nstacked_frames  : stackSize (int, number of frames to stack)
    shrink_rows      : shrinkRows (int, x size of shrunk frame)
    shrink_cols      : shrinkCols (int, y size of shrunk frame)
    pretrain_len     : preTrainLen (int, number of experiences to initially fill mem with)
    save_path        ; saveFilePath (string, path of checkpoint and train params file)
    save_period      : savePeriod (int, save model every savePeriod episodes)
    restart_training : restartTraining (int, if 1 start from beginning. If 0, cont)
    ckpt_file        : ckptFile (string, name of the checkpoint file to use when saving)
    test_flag        : testFlag (bool, whether or not to test the agent)
    env_name         : envName (string, name of the gym environment (game) to use)
    architecture     : architecture (string, the network architecture to use)
    fixed_Q          : fixedQ (int, 0 if not using, 1 if using the fixed-Q technique)
    fixed_Q_steps    : fixedQSteps (int, steps between weights copies w/ fixed-Q)
    double_dqn       : doubleDQN (int, 0 if not using, 1 if using double DQN technique)
    per              : per (int, 1 if using prioritized experience replay, 0 if not)
    per_a            : perA (float, the alpha parameter in eq. 1 of Schaul16)
    per_b            : perB (float, the beta parameter in IS weights of Schaul16)
    per_e            : perE (float, the epsilon parameter in prop. prior. of Schaul16)
    per_b_anneal     : perBAnneal (float, annealment rate of IS weights)
    """
    # Assume the file is in the current working directory
    fname = os.path.join(os.getcwd(), fname)
    # Make sure the file exists
    if not os.path.isfile(fname):
        raise IOError
    # Set up registers for casting to different data types
    type_register = param_file_registers()
    # Read file
    hyperparams = {}
    with open(fname, 'r') as f:
        for line in f:
            key, value = line.split(':')
            key = key.strip()
            # Cast the parameter to the appropriate type
            if key in type_register['floats']:
                value = float(value)
            elif key in type_register['ints']:
                value = int(value)
            elif key in type_register['strings']:
                value = value.strip()
            else:
                print('Hyperparameter not found!')
                raise IOError
            hyperparams[key] = value
    # Convert from int to bool where appropriate (see TODO)
    if hyperparams['restart_training'] == 1:
        hyperparams['restart_training'] = True
    else:
        hyperparams['restart_training'] = False
    # Check to make sure the architecture has been defined
    if hyperparams['architecture'] not in archRegister:
        raise ValueError("Error, unrecognized network architecture!")
    # Check for option conflicts
    if check_agent_option_conflicts(hyperparams) == 0:
        sys.exit(0)
    return hyperparams



#============================================
#                 crop_frame
#============================================
def crop_frame(frame, crop):
    """
    This function handles the different cases for cropping the frame to the proper
    size. It doesn't matter whether crop[0] and/or crop[2] are zero or not because the
    first term in a slice is always included, whereas the last one is not.

    Parameters:
    -----------
        frame : ndarray
            The game frame

        crop : tuple
            The number of rows to chop off from the top and bottom and number of columns
            to chop off from the left and right.

    Returns:
    --------
        cropFrame : ndarray
            The cropped version of frame
    """
    cropFrame = None
    # Sanity check (this part could go in another function that gets called initially,
    # since it only needs to happen once)
    if (crop[0] >= frame.shape[0]) or (crop[1] >= frame.shape[0]):
        raise ValueError("Error, can't crop more rows than are in frame!")
    if (crop[2] >= frame.shape[1]) or (crop[3] >= frame.shape[1]):
        raise ValueError("Error, can't crop more cols than are in frame!")
    if crop[0] + crop[1] >= frame.shape[0]:
        raise ValueError("Error, total crop from bot and top too big!")
    if crop[2] + crop[3] >= frame.shape[1]:
        raise ValueError("Error, total crop from left and right too big!")
    if (crop[1] != 0) and (crop[3] != 0):
        cropFrame = frame[crop[0]:-crop[1], crop[2]:-crop[3]]
    elif (crop[1] == 0) and (crop[3] != 0): 
        cropFrame = frame[crop[0]:, crop[2]:-crop[3]]
    elif (crop[1] == 0) and (crop[3] == 0):
        cropFrame = frame[crop[0]:, crop[2]:]
    elif (crop[1] != 0) and (crop[3] == 0):
        cropFrame = frame[crop[0]:-crop[1], crop[2]:]
    # Sanity check
    if cropFrame is None:
        raise ValueError("Error in crop_frame, cropFrame not set!")
    if (sum(crop) != 0) and (cropFrame.shape == frame.shape):
        raise ValueError("Error in crop_frame, shapes equal when they shouldn't be!")
    elif (sum(crop) == 0) and (cropFrame.shape != frame.shape):
        raise ValueError("Error in crop_Frame, shapes not equal when they should be!") 
    return cropFrame



#============================================
#             preprocess_frame
#============================================
def preprocess_frame(frame, crop, shrink):
    """
    This function handles grayscaling the frame and cropping the frame to the proper size

    Parameters:
    -----------
        frame : ndarray
            The game image to crop and grayscale

        crop : tuple
            The number of rows to chop off both the top and bottom, the number of cols to
            chop off both the left and right. (top, bot, left, right)

        shrink : tuple
            The (x,y) size of the shrunk image

    Returns:
    --------
        processed_frame : ndarray
            The grayscaled and cropped frame
    """
    # Grayscale the image
    frame = skimage.color.rgb2grey(frame)
    # Crop the image. We don't need blank space or things on the screen that aren't game
    # objects. The crop tuple contains the number of pixels to chop off each dimension.
    # This poses a problem when that number is 0 (e.g. the slice 0:0 will create an empty
    # slice).
    frame = crop_frame(frame, crop)
    # Normalize the image
    frame = frame / 255.
    # To reduce the computational complexity, we can shrink the image
    frame = skimage.transform.resize(frame, [shrink[0], shrink[1]])
    return frame



#============================================
#               stack_frames
#============================================
def stack_frames(frame_stack, state, new_episode, stack_size, crop, shrink):
    """
    This function does two things: it takes in the current state and preprocesses it.
    Then, it adds the processed frame to the stack of frames. Two versions of this stack
    are returned: a tensorial version (ndarray) and a deque for easy pushing and popping.

    Parameters:
    -----------
        frame_stack : deque
            The deque version of the stack of processed frames

        state : gym state
            This is effectively the raw frame from the game

        new_episodes : bool
            If True, then we need to produce a clean deque and tensor. Otherwise, we can
            just add the given frame (state) to the stack

        stack_size : int
            The number of frames to include in the stack

        crop : tuple
            (top, bot, left, right) to chop off each edge of the frame

        shrink : tuple
            (x,y) size of the shrunk frame

    Returns:
    --------
        stacked_state : ndarray
            This is the tensorial version of the frame stack deque

        frame_stack : deque
            The deque version of the stack of frames
    """
    # Error check
    if (new_episode is False) and (isinstance(frame_stack, collections.deque) is False):
        print('Error, must pass existing frame stack if not starting a new episode!')
        sys.exit()
    # Preprocess the given state
    state = preprocess_frame(state, crop, shrink)
    # Start fresh if this is a new episode
    if new_episode:
        frame_stack = collections.deque([state for i in range(stack_size)],
                                        maxlen=stack_size)
    # Otherwise, add the frame to the stack
    else:
        frame_stack.append(state)
    # Create the tensorial version of the stack
    stacked_state = np.stack(frame_stack, axis=2)
    return stacked_state, frame_stack



#============================================
#               Memory Class
#============================================
class Memory():
    """
    This class holds and manages the experience buffer for DQNs.

    Parameters:
    -----------
        max_size : int
            The max number of experience tuples the buffer can hold before it "forgets"

        pretrain_len : int
            The number of initial, samply/dummy experiences to full the buffer with so
            we don't run into the empty memory problem when trying to train initially

        env : gym environment
            The environment for the game. Used to pre-populate the memory buffer

        stack_size : int
            Number of frames to stack

        crop : tuple
            (top, bot, left, right) to chop off each edge of the frame

        shrink : tuple
            (x,y) size of the shrunk frame
    """
    #-----
    # Constructor
    #-----
    def __init__(self, max_size, pretrain_len, env, stack_size, crop, shrink):
        self.max_size = max_size
        self.pretrain_len = pretrain_len
        self.buffer = collections.deque(maxlen = self.max_size)
        self.pre_populate(env, stack_size, crop, shrink)

    #-----
    # Pre-Populate
    #-----
    def pre_populate(self, env, stack_size, crop, shrink):
        """
        This function initially fills the experience buffer with sample experience
        tuples to avoid the empty memory problem.

        Parameters:
        -----------
            env : gym environment
                The environment for the game that's being learned by the agent

            stack_size : int
                The number of frames to stack together for temporal differencing

            crop : tuple
                (top, bot, left, right) to chop off each edge of the frame

            shrink : tuple
                (x, y) size of the shrunk frame

        Returns:
        --------
            None
        """
        # Get initial state
        state = env.reset()
        # Process and stack initial frames
        state, frame_stack = stack_frames(None, state, True, stack_size, crop, shrink)
        # Loop over the desired number of sample experiences
        for i in range(self.pretrain_len):
            # Choose a random action. randint chooses in [a,b)
            action = np.random.randint(0, env.action_space.n)
            # Take action
            next_state, reward, done, _ = env.step(action)
            # Add next state to stack of frames
            next_state, frame_stack = stack_frames(frame_stack, next_state, False,
                                                        stack_size, crop, shrink)
            # Add experience to memory
            self.add((state, action, reward, next_state, done))
            # If we're in a terminal state, we need to reset things
            if done:
                state = env.reset()
                state, frame_stack = stack_frames(None, state, True, stack_size, crop,
                                                  shrink)
            # Otherwise, update the state and continue
            else:
                state = next_state

    #-----
    # Add
    #-----
    def add(self, experience):
        """
        This function just adds the newest experience tuple to the buffer

        Parameters:
        -----------
            experience : tuple
                Contains the state, action, reward, next_state, and done flag

        Returns:
        --------
            None 
        """
        self.buffer.append(experience)

    #-----
    # Sample
    #-----
    def sample(self, batch_size):
        """
        This function returns a randomly selected subsample of size batch_size from
        the buffer. This subsample is used to train the DQN. Note that a deque's size is
        determined only from the elements that are in it, not from maxlen. That is, if
        you have a deque with maxlen = 10, but only one element has been added to it,
        then it's size is actually 1.

        Parameters:
        -----------
            batch_size : int
                The size of the sample to be returned

        Returns:
        --------
            sample : list
                A list of randomly chosen experience tuples from the buffer. Chosen
                without replacement. The length of the list is batch_size
        """
        # Choose random indices from the buffer
        # Make sure the batch_size isn't larger than the current buffer size or np will
        # complain
        try:
            indices = np.random.choice(np.arange(len(self.buffer)),
                                        size = batch_size,
                                        replace = False)
        except ValueError:
            raise("Error, need batch_size < buf_size when sampling from memory!")
        return [self.buffer[i] for i in indices]



#============================================
#             save_train_params 
#============================================
def save_train_params(decay, rewards, mem, path, qstep):
    """
    This function saves the crucial training parameters needed in order to continue
    where training left off.

    Parameters:
    -----------
        ep : int
            The most recent episode to have finished

        decay : int
            The value of the decay_step used in explore-exploit epsilon greedy

        rewards : list
            List of the total reward earned for each completed episode

        mem : deque
            The memory buffer for the current training session

        path : string
            Place to save this information

        qstep : int
            The current step that we're on with regards to when the targetQNet should
            be updated. Only matters if using fixed-Q.

    Returns:
    --------
        None
    """
    # Episode, decay, and episode rewards
    with open(os.path.join(path, 'ep_decay_reward.txt'), 'w') as f:
        f.write(str(decay) + '\n')
        f.write(str(qstep) + '\n')
        for i in range(len(rewards)):
            f.write(str(rewards[i]) + '\n')
    # States
    states = np.array([s[0] for s in mem], ndmin=3)
    np.savez(os.path.join(path, 'exp_states'), *states)
    # Actions
    actions = np.array([s[1] for s in mem])
    np.savez(os.path.join(path, 'exp_actions'), *actions)
    # Rewards
    exp_rewards = np.array([s[2] for s in mem])
    np.savez(os.path.join(path, 'exp_rewards'), *exp_rewards)
    # Next states
    next_states = np.array([s[3] for s in mem], ndmin=3)
    np.savez(os.path.join(path, 'exp_next_states'), *next_states)
    # Dones
    dones = np.array([s[4] for s in mem])
    np.savez(os.path.join(path, 'exp_dones'), *dones)



#============================================
#             load_train_params
#============================================
def load_train_params(path, max_len):
    """
    This function reads in the data saved to the files produced in save_train_params
    so that training can continue where it left off.

    Parameters:
    -----------
        path : string
            The path to the required data files

        max_len : int
            The maximum length of the memory buffer

    Returns:
    --------
        train_params : tuple
            (start_episode, decay_step, totalRewards, memory, qstep)
    """
    # Read the ep_decay_reward file
    with open(os.path.join(path, 'ep_decay_reward.txt'), 'r') as f:
        # Decay step
        decay_step = int(f.readline())
        qstep = int(f.readline())
        # Episode rewards
        ep_rewards = []
        for line in f:
            ep_rewards.append(float(line))
    # Get the most recently finished episode
    ep = len(ep_rewards)
    # Load the states, actions, rewards, next_states, and dones arrays
    states = np.load(os.path.join(path, 'exp_states.npz'))
    actions = np.load(os.path.join(path, 'exp_actions.npz'))
    rewards = np.load(os.path.join(path, 'exp_rewards.npz'))
    next_states = np.load(os.path.join(path, 'exp_next_states.npz'))
    dones = np.load(os.path.join(path, 'exp_dones.npz'))
    # Sanity check
    nstates = len(states.files)
    if len(actions.files) != nstates or\
        len(rewards.files) != nstates or\
        len(next_states.files) != nstates or\
        len(dones.files) != nstates:
        print('Error, length of read in states array does not match length of actions, '
                'rewards, next_states, or dones!')
        sys.exit()
    # Get experience tuples to fill mem buffer (state, action, reward, next_state, done)
    buf = collections.deque(maxlen=max_len)
    for i in range(nstates):
        key = 'arr_' + str(i)
        exp = (states[key], actions[key], rewards[key], next_states[key], dones[key])
        buf.append(exp)
    # Package everything up
    train_params = (ep, decay_step, ep_rewards, buf, qstep)
    return train_params



#============================================
#               Sumtree Class
#============================================
class Sumtree():
    """
    Prioritized experience replay makes use of a sum tree to efficiently store and fetch
    data. A sum tree is a binary tree where the value of each node is the sum of the
    values in its child nodes. Here, the actual priorities are stored in the leaf nodes
    of the tree. This is an unsorted tree.
    
    Assuming a perfectly balanced tree, the number of nodes in the tree is 
    nNodes = 2 * nLeafs - 1. This is because, in a binary tree, the number of nodes at
    a given level is twice the number of nodes in the level before it. We then have to
    subract off 1 because at the root level there is only one node. This assumes a 
    perfectly balanced tree (that is, every node has both a left and right child and
    that each subtree descends to the same level), i.e.,

                                    o
                                   / \
                                  o   o
                                 / \ / \
                                o  o o  o

    The nodes are stored in an array level-wise. That is, root is index 0, root's left
    child is at index 1, root's right child is at index 2, then we go to the next level
    down and go across left to right. As such, the indices of the leaf nodes start at
    nLeafs - 1 and continue on to the end of the array. 

    Parameters:
    -----------
        nLeafs : int
            The number of leaf nodes the tree will have. This is equal to the number of
            experiences we want to store, since the priorities for each experience go
            into the leaf nodes.
`
    Attributes:
    -----------
        dataPointer : int
            The leaves of the tree are filled from left to right. This is an index that
            keeps track of where we are in the leaf row of the tree.

        tree : ndarray
            This is an array used to store the actual sum tree.

        data : ndarray
            Attached to each leaf is a priority (stored in the tree attribute) as well as
            the actual experience that has that priority. This array holds the experience
            tuples.

    Methods:
    --------
    """
    #-----
    # Constructor
    #-----
    def __init__(self, nLeafs):
        self.nLeafs = nLeafs
        self.dataPointer = 0
        self.tree = np.zeros(2 * self.nLeafs - 1)
        self.data = np.zeros(self.nLeafs, dtype=object)

    #-----
    # add
    #-----
    def add(self, data, priority):
        """
        This function takes in an experience as well as the priority assigned to that
        experience, and assigns it to a leaf node in the tree, propagating the the
        changes throughout the rest of the tree.

        Parameters:
        -----------
            priority : float
                The priority that has been assigned to this particular experience

            data : tuple
                The experience tuple being added to the tree

        Returns:
        --------
            None
        """
        # Get the index of the array corresponding to the current leaf node
        tree_index = self.dataPointer + self.nLeafs - 1
        # Insert the experience at this location (like the deque, this starts to 'forget'
        # experiences once the max capacity is reached, i.e., they are overwritten)
        self.data[self.dataPointer] = data
        # Update the tree
        self.update(tree_index, priority)
        # Advance to the next leaf
        self.dataPointer += 1
        # If we're above the max value, then we go back to the beginning
        if self.dataPointer >= self.nLeafs:
            self.dataPointer = 0

    #-----
    # update
    #-----
    def update(self, tree_index, priority):
        """
        This function handles updating the current leaf's priority score and then
        propagating that change throughout the rest of the tree.

                                            0
                                           / \
                                          1   2
                                        /  \  / \
                                       3   4  5  6

        The numbers above are the indices of the nodes. If the tree looks like:

                                            48
                                           / \
                                          13   35
                                        /  \  / \
                                       4   9  33  2

        and the value of node 6 changes to 8, the new sumtree will look like:

                                            54
                                           / \
                                          13   41
                                        /  \  / \
                                       4   9  33  8

        That is, we need to update the value of each of the changed node's ancestors.
        We get at the parent node by doing (currentNodeIndex - 1) // 2. E.g.,
        (6-1) // 2 = 2, then (2-1)//2 = 0, which gives us the indices of the two nodes
        we would need to update in this case (2 and 0). The update is to simply add the
        change made to the current node to the value of it's parent. E.g., here node 6
        has change = 8 - 2 = 6, so the update to node 2 is: change2 = 35 + change = 41.
        We then repeat this process until we've updated root.

        Parameters:
        -----------
            tree_index : int
                The index of self.tree that corresponds to the current leaf node

            priority : float
                The value of the priority to assign to the current leaf node

        Returns:
        --------
            None
        """
        # Get the difference between the new priority and the old priority
        deltaPriority = priority - self.tree[tree_index]
        # Update the node with the new priority
        self.tree[tree_index] = priority
        # Propogate the change throughout the rest of the tree (we're done after we
        # update root, which has an index of 0)
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += deltaPriority

    #-----
    # get_leaf
    #-----
    def get_leaf(self, value):
        """
        This function returns the experience whose priority is the closest to the passed
        value.

                                            48
                                           / \
                                          13   35
                                        /  \  / \
                                       4   9  33  2

        For the above tree (the numbers are the priorities, not indices), let's say we're
        in the situation where batchSize = 6, so we've broken up the range [0,48] into 6
        equal ranges. If we're on the first one, it will span [0,8). We choose a random
        number from there, which is value, so let's say we get value = 7. This function
        would then return node 4, which has a value of 9 and is the closest to the passed
        value of 7.

        Parameters:
        -----------
            value : float
                We want to find the experience whose priority is the closest to this
                number

        Returns:
        --------
            index : int
                The index of the tree corresponding to the chosen experience

            priority : float
                The priority of the chosen experience

            experience : tuple
                The experience whose priority is closest to value
        """
        # Start at root
        parentIndex = 0
        # Search the whole tree
        while True:
            # Get the indices of the current node's left and right children
            leftIndex = 2 * parentIndex + 1
            rightIndex = leftIndex + 1
            # Check exit condition
            if leftIndex > len(self.tree):
                leafIndex = parentIndex
                break
            # Otherwise, continue the search
            else:
                if value <= self.tree[leftIndex]:
                    parentIndex = leftIndex
                else:
                    value -= self.tree[leftIndex]
                    parentIndex = rightIndex
        # Get the experience corresponding to the selected leaf
        dataIndex = leafIndex - self.nLeafs + 1
        return leafIndex, self.tree[leafIndex], self.data[dataIndex]

    #-----
    # total_priority
    #-----
    @property
    def total_priority(self):
        """
        This function returns the root node of the tree, which is just the sum of all of
        the priorities. The property decorator lets us access it as if it were an
        attribute.

        Parameters:
        -----------
            None

        Returns:
        --------
            totalPriority : float
                The sum of each leaf's priority in the tree, which is just the root node
                since this is a sum tree
        """
        return self.tree[0]



#============================================
#           PriorityMemory Class
#============================================
class PriorityMemory(Memory):
    """
    This class serves as the memory buffer in the case that prioritized experience
    replay is being used. It functions in much the same way as Memory(), but employs a
    SumTree() instead of a deque. This means that the adding and sampling methods are
    different.

    Parameters:
    -----------
        max_size : int
            The max number of experience tuples the buffer can hold before it begins to
            "forget" experiences (i.e., they are overwritten)

        pretrain_len : int
            The number of initial, samply/dummy experiences to full the buffer with so
            we don't run into the empty memory problem when trying to train initially

        env : gym environment
            The environment for the game. Used to pre-populate the memory buffer

        stack_size : int
            Number of frames to stack

        crop : tuple
            (top, bot, left, right) to chop off each edge of the frame

        shrink : tuple
            (x,y) size of the shrunk frame

        perParams : list
            A list of alpha, beta, epsilon, and the annealment rate

    Methods:
    --------
        pass

    Attributes:
    -----------
        pass
    """
    #-----
    # Constructor
    #-----
    def __init__(self, max_size, pretrain_len, env, stack_size, crop, shrink, perParams):
        # Call parent's constructor
        super().__init__(max_size, pretrain_len, env, stack_size, crop, shrink)
        # Set per parameters
        self.perA = perParams[0]
        self.perB = perParams[1]
        self.perBAnneal = perParams[2]
        self.perE = perParams[3]
        # Overload the buffer
        self.buffer = SumTree(self.max_size)
        self.upperPriority = 1.0

    #-----
    # Add
    #-----
    def add(self, experience):
        """
        This function stores the newest experience tuple, along with a priority, to the
        buffer. According to Schaul16 algorithm 1, the new experiences are added with a
        priority equal to the current max priority in the tree.

        Parameters:
        -----------
            experience : tuple
                Contains the state, action ,reward, next_state, and done flag

        Returns:
        --------
            None
        """
        # Get the current max priority in the tree. Recall that the left nodes hold the
        # priority and that they are stored as the last max_size elements in the array
        # that holds the tree
        maxPriority = np.max(self.buffer.tree[-self.buffer.max_size:])
        # If the maxPriority is 0, then we need to set it to the predefined upperPriority
        # because a priority of 0 means that the experience will never be chosen; and we
        # want every experience to have a chance at being chosen.
        if maxPriority == 0:
            maxPriority = self.upperPriority
        self.buffer.add(experience, maxPriority)

    #-----
    # Sample
    #-----
    def sample(self, batchSize):
        """
        This function returns a subsample of experiences from the memory buffer to be used
        in training. The probability for a particular experience to be chosen is given by
        equation 1 in Schaul16. The details of how to sample from the sumtree are given in
        Appendix B.2.1: Proportional prioritization in Schaul16. Essentially, we break up
        the range [0, priority_total] into batchSize segments of equal size. We then
        uniformly choose a value from each segment and get the experiences that correspond
        to each of these sampled values.

        Parameters:
        -----------
            batchSize : int
                The size of the sample to return

        Returns:
        --------
            indices : ndarray
                An array of tree indices corresponding to the sampled experiences

            experiences : list
                A list of batchSize experiences chosen with probabilities given by
                Schaul16 equation 1

            isWeights : ndarray
                An array containing the IS weights for each sampled experience
        """
        # We need to return the selected samples (to be used in training), the indices
        # of these samples (so that the tree can be properly updated), and the importance
        # sampling weights to be used in training
        indices = np.zeros((batchSize,))
        priorities = np.zeros((batchSize, 1))
        experiences = []
        # We need to break up the range [0, p_tot] equally into batchSize segments, so
        # here we get the width of each segment
        segmentWidth = self.buffer.total_priority / batchSize
        # Anneal the strength of the IS weights (cap the parameter at 1)
        self.perB = np.min([1., self.perB + self.perBAnneal])
        # Loop over the desired number of samples
        for i in range(batchSize):
            # We need to uniformly select a value from each segment, so here we get the
            # lower and upper bounds of the segment
            lowerBound = i * segmentWidth
            upperBound = (i + 1) * segmentWidth
            # Choose a value from within the segment
            value = np.random.uniform(lowerBound, upperBound)
            # Retrieve the experience whose priority matches value from the tree
            index, priority, experience = self.buffer.get_leaf(value)
            indices[i] = index
            priorities[i, 0] = priority
            experiences.append(experience)
        # Calculate the importance sampling weights
        samplingProbabilities = priorities / self.buffer.total_priority
        isWeights = np.power(batchSize * samplingProbabilities, -self.perB)
        isWeights = isWeights / np.max(isWeights)
        return indices, experiences, isWeights

    #-----
    # Update
    #-----
    def update(self, indices, errors):
        """
        This function uses the new errors generated from training in order to update the
        priorities for those experiences that were selected in sample().

        Parameters:
        -----------
            indices : ndarray
                Array of tree indices corresponding to those experiences used in the
                training batch

            errors : ndarray
                Array of TD errors for the chosen experiences

        Returns:
        --------
            None
        """
        # Calculate priorities from errors (proportional prioritization)
        priorities = np.abs(errors) + self.perE
        # Clip the errors for stability
        priorities = np.min([priorities, self.upperPriority])
        # Apply alpha
        priorities = np.power(priorities, self.perA)
        # Update the tree
        for ind, p in zip(indices, priorities):
            self.buffer.update(ind, p)
