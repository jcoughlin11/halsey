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
                    'discount']
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
                    'restart_training']
    string_params = ['save_path',
                     'ckpt_file']
    type_register = {'floats' : float_params,
                    'ints' : int_params,
                    'strings' : string_params}
    return type_register



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
    if hyperparams['restart_training'] == 1:
        hyperparams['restart_training'] = True
    else:
        hyperparams['restart_training'] = False
    return hyperparams



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
    # objects
    frame = frame[crop[0]:-crop[1], crop[2]:-crop[3]]
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
                state, frame_stack = ntack_frames(None, state, True, stack_size, crop,
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
def save_train_params(ep, decay, rewards, mem, path):
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

    Returns:
    --------
        None
    """
    # Episode, decay, and episode rewards
    with open(os.path.join(path, 'ep_decay_reward.txt'), 'w') as f:
        # The +1 is because episode is indexed from 0, so in order to have it match with
        # len(ep_rewards) when loading, I add the +1
        f.write(str(ep + 1) + '\n')
        f.write(str(decay) + '\n')
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
            (start_episode, decay_step, totalRewards, memory)
    """
    # Read the ep_decay_reward file
    with open(os.path.join(path, 'ep_decay_reward.txt'), 'r') as f:
        # Get episode number. The +1 is because what's written is the most recently
        # completed episode, so if we want to start at the next one, we add 1
        ep = int(f.readline()) + 1
        # Decay step
        decay_step = int(f.readline())
        # Episode rewards
        ep_rewards = []
        for line in f:
            ep_rewards.append(float(line))
        # Sanity check
        if (len(ep_rewards) != (ep - 1)):
            print('Error, number of episode rewards does not match episode number!')
            sys.exit()
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
        exp = (states[i], actions[i], rewards[i], next_states[i], dones[i])
        buf.append(exp)
    # Package everything up
    train_params = (ep, decay_step, ep_rewards, buf)
    return train_params
