"""
Title: io.py
Author: Jared Coughlin
Date: 7/30/19
Purpose: Contains tools related to reading and writing files to disk
Notes:
    * https://tinyurl.com/y2nhqrce (good h5py guide)
    * http://docs.h5py.org/en/stable/special.html (h5py special dtpyes)
    * compression_opts goes up to 9
    * The higher the level, the more aggressive the comp, but the more
    processor intensive it is
    * The default is 4, which I've included just to remind myself the
    option exists in case I don't remember to look at this header
"""
import collections
import os

import h5py
import numpy as np
import subprocess32

import nnetworks as nw
import nnutils as nu


# Registers
archRegister = ["conv1",
    "dueling1",
    "perdueling1",
    "rnn1"
]
lossRegister = ['mse',
    'per_mse',
]
optimizerRegister = ['adam',
]


# ============================================
#           param_file_registers
# ============================================
def param_file_registers():
    """
    Sets up the registers containing the types associated with each
    parameter.

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
    float_params = [
        "discount",
        "eps_decay_rate",
        "epsilon_start",
        "epsilon_stop",
        "learning_rate",
        "per_a",
        "per_b",
        "per_b_anneal",
        "per_e",
    ]
    int_params = [
        "batch_size",
        "crop_bot",
        "crop_left",
        "crop_right",
        "crop_top",
        "enable_double_dqn",
        "enable_fixed_Q",
        "enable_per",
        "fixed_Q_steps",
        "max_episode_steps",
        "memory_size",
        "n_episodes",
        "n_stacked_frames",
        "pretrain_len",
        "pretrain_max_ep_len",
        "render_flag",
        "shrink_cols",
        "shrink_rows",
        "save_period",
        "test_flag",
        "trace_len",
        "time_limit",
        "train_flag",
    ]
    string_params = [
        "architecture",
        "ckpt_file",
        "env_name",
        'loss',
        'optimizer',
        "save_path"
    ]
    type_register = {
        "floats": float_params,
        "ints": int_params,
        "strings": string_params,
    }
    return type_register


# ============================================
#        check_agent_option_conflicts
# ============================================
def check_agent_option_conflicts(params):
    """
    Checks for any conflicts between techniques that have been turned on
    in the parameter file.

    Parameters:
    -----------
        params : dict
            The parameters read in from the parameter file. It was
            easier to pass the whole dict rather than specific options
            since I don't yet know all of the options that will be
            included.

    Raises:
    -------
        pass

    Returns:
    --------
        conflict_flag : int
            If 0, a conflict was found and the code will abort. The
            error message is generated nad printed from within this
            function. If 1, then everything is fine (hopefully!).
    """
    # Check to make sure the architecture has been defined
    if params["architecture"] not in archRegister:
        raise ValueError("Error, unrecognized network architecture!")
    # Check for valid loss function
    if params['loss'] not in lossRegister:
        raise ValueError("Error, unrecognized loss function!")
    # Check for valid optimizer function
    if params['optimizer'] not in optimizerRegister:
        raise ValueError("Error, unrecognized optimizer function!")
    # Double DQN requires fixed-Q
    if params["enable_double_dqn"] and not params["enable_fixed_Q"]:
        raise ValueError("Error, double dqn requires the use of fixed Q!")
    # Make sure the save path exists. If it doesn't, try and make it
    if not os.path.exists(params["save_path"]):
        os.path.makedirs(params["save_path"])
    # If it does exist, make sure it's a directory
    elif not os.path.isdir(params["save_path"]):
        raise ValueError("savePath exists but is not a dir!")
    # Make sure either the train flag or test flag (or both) are set
    if not params['train_flag'] and not params['test_flag']:
        raise ValueError("Error, neither training nor testing enabled!")


#============================================
#              read_param_file
#============================================
def read_param_file(fname):
    """
    Reads the data from the parameter file.

    architecture        : string, the network architecture to use
    batch_size          : int, size of batches used for training
    ckpt_file           : string, name of file to use for saving/loading
    crop_bot            : int, num rows to chop off bottom of frame
    crop_left           : int, num cols to chop off left of frame
    crop_right          : int, num cols to chop off right of frame
    crop_top            : int, num rows to chop off top of frame
    discount            : float, reward discount rate
    enable_double_dqn   : int, 0 if not using, 1 if using double DQN
    enable_fixed_Q      : int, 0 if not using, 1 if using fixed-Q
    enable_per          : int, 1 if using prioritized experience replay,
                          0 if not
    env_name            : string, name of the gym environment to use
    eps_decay_rate      : float, rate the  explore-exploit param decays
    epsilon_start       : float, start val of explore-exploit parameter
    epsilon_stop        : float, min value of explore-exploit parameter
    fixed_Q_steps       : int, steps between weight copies w/ fixed-Q
    learning_rate       : float, network learning rate
    loss                : string, the loss function to minimize
    max_episode_steps   : int, max number of steps per episode
    memory_size         : int, max number of experiences to store in
                          memory buffer
    metrics             : list, the quantities to track during training
    n_episodes          : int, number of episodes to train for
    n_stacked_frames    : int, number of frames to stack
    optimizer           : string, name of the optimizer to use
    per_a               : float, alpha parameter in eq. 1 of Schaul16
    per_b               : float, beta param in IS weights of Schaul16
    per_b_anneal        : float, annealment rate of IS weights
    per_e               : float, epsilon parameter in prop. prior. of
                          Schaul16
    pretrain_len        : int, num experiences to initially fill mem
    pretrain_max_ep_len : int, max ep length when filling mem buffer
    render_flag         : int, if 1, render scene during testing
    save_path           ; string, path of checkpoint and param file
    save_period         : int, save model every savePeriod episodes
    shrink_cols         : int, y size of shrunk frame
    shrink_rows         : int, x size of shrunk frame
    test_flag           : int, if 1, test the agent
    trace_len           : int, num connected frames in RNN sample
    time_limit          : int, max number of seconds to run for
    train_flag          : int, if 1, train network, if 0, load saved

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
    # Set up registers for casting to different data types
    type_register = param_file_registers()
    # Read file
    hyperparams = {}
    with open(fname, "r") as f:
        for line in f:
            # Skip lines that begin with '#' and empty lines
            if line[0] == '#' or not line.strip():
                continue
            try:
                key, value = line.split(':')
            except ValueError:
                print("Error, couldn't parse line '{}' in param "
                    "file!".format(line)
                )
                sys.exit(1)
            key = key.strip()
            # Cast the parameter to the appropriate type
            if key in type_register["floats"]:
                value = float(value)
            elif key in type_register["ints"]:
                value = int(value)
            elif key in type_register["strings"]:
                value = value.strip()
            else:
                print("Hyperparameter {} not found!".format(key))
                raise IOError
            hyperparams[key] = value
    return hyperparams


# ============================================
#              get_hyperparams
# ============================================
def get_hyperparams(fname, continueFlag):
    """
    Reads in the parameter file that contains the network's
    hyperparameters. The layout is:

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
    # Make sure the file exists
    if not os.path.isfile(fname):
        raise FileNotFoundError
    # Read file
    hyperparams = read_param_file(fname)
    # If we're restarting, read the saved version of the original
    # parameter file in savePath to minimize the chances of diffs
    # between the just-read param file and the original, saved one
    if continueFlag:
        fname = os.path.join(hyperparams['save_path'], 'dqn_hyperparams.txt')
        print("Continuing training...")
        hyperparams = read_param_file(fname)
    else:
        print("Restarting training...")
    # Check for option conflicts
    check_agent_option_conflicts(hyperparams)
    # Copy the parameter file to the save path for use with restarting
    # and as a record. Only need to do this if it hasn't already been
    # saved
    if not continueFlag:
        backup_param_file(fname, hyperparams["save_path"])
    # Set the loss and optimizer functions to their proper values based
    # on the string representation
    # Add the continue flag to the hyperparams so the agent knows what
    # to do
    hyperparams['restart_training'] = 0 if continueFlag else 1
    hyperparams = nu.set_loss(hyperparams)
    hyperparams = nu.set_optimizer(hyperparams)
    return hyperparams


#============================================
#            backup_param_file
#============================================
def backup_param_file(fname, savePath):
    """
    Makes a copy of the parameter file in the savePath directory. This
    copy is used for restarting training (it serves as a guard against
    changes being made to the original param file between stopping and
    restarting training) and it serves as a record for what parameters
    were used for a particular model once training is complete.

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
    backupFile = os.path.join(savePath, os.path.basename(fname))
    subprocess32.call(['cp', fname, backupFile])


#============================================
#                save_memory
#============================================
def save_memory(memBuffer, savePath):
    """
    Saves the contents of the memory buffer to an hdf5 file.

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
    # Case 1: memBuffer is a deque
    if isinstance(memBuffer, collections.deque):
        save_deque_memory(memBuffer, savePath)
    # Case 2: memBuffer is a SumTree. In this case, the whole data
    # structure needs to be saved
    elif isinstance(memBuffer, nu.SumTree):
        save_sumtree_memory(memBuffer, savePath)
    # Unrecognized case
    else:
        raise TypeError("Error, unrecognized memory buffer type!")


#============================================
#             save_deque_memory
#============================================
def save_deque_memory(memBuffer, savePath):
    """
    Handles saving the agent's memories when the memory buffer is a
    deque.

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
    # Create hdf5 file
    with h5py.File(os.path.join(savePath, 'memory_buffer.h5'), 'w') as h5f:
        # Create the empty datasets to store the memory components in
        # a group named after the memory type. This makes reading the
        # data back in easier because type(memBuffer) can be identified
        # without passing any flags
        nSamples = len(memBuffer)
        stateShape = [nSamples] + list(memBuffer[0][0].shape)
        g = h5f.create_group('deque')
        g.create_dataset('states', shape=stateShape, compression='gzip', compression_opts=4, dtype=np.float)
        g.create_dataset('actions', (nSamples,), compression='gzip', compression_opts=4, dtype=np.int)
        g.create_dataset('rewards', (nSamples,), compression='gzip', compression_opts=4, dtype=np.float)
        g.create_dataset('next_states', shape=stateShape, compression='gzip', compression_opts=4, dtype=np.float)
        g.create_dataset('dones', (nSamples,), compression='gzip', compression_opts=4, dtype=np.int)
        # Loop over each sample in the buffer
        for i, sample in enumerate(memBuffer):
            g['states'][i] = sample[0]
            g['actions'][i] = sample[1]
            g['rewards'][i] = sample[2]
            g['next_states'][i] = sample[3]
            g['dones'][i] = sample[4]


#============================================
#            save_sumtree_memory
#============================================
def save_sumtree_memory(memBuffer, savePath):
    """
    Handles saving the agent's memories when the memory buffer is a
    sum tree.

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
    # Create hdf5 file
    with h5py.File(os.path.join(savePath, 'memory_buffer.h5'), 'w') as h5f:
        # Create a group named after the buffer type (sumtree) to make
        # loading easier (see save_deque_memory)
        g = h5f.create_group('sumtree')
        # Create datasets for the counters and tree array
        data = [memBuffer.nLeafs, memBuffer.dataPointer]
        g.create_dataset('counters', data=data)
        g.create_dataset('tree', data=memBuffer.tree)
        # The data attribute of the sum tree is where the actual
        # experiences are stored, so it's an array of tuples. This
        # means it's easier to package everything into another group
        # in the same manner as in save_deque_memory
        dg = g.create_group('data')
        stateShape = [memBuffer.nLeafs] + list(memBuffer.data[0][0].shape)
        dg.create_dataset('states', shape=stateShape, dtype=np.float)
        dg.create_dataset('actions', (memBuffer.nLeafs,), dtype=np.int)
        dg.create_dataset('rewards', (memBuffer.nLeafs,), dtype=np.float)
        dg.create_dataset('next_states', shape=stateShape, dtype=np.float)
        dg.create_dataset('dones', (memBuffer.nLeafs,), dtype=np.int)
        # Loop over each sample in the buffer. If the buffer is not yet
        # full then it will be comprised of both tuples and zeros
        # This is not a particularly good way of doing this, I just
        # wanted to get it running to test PER and I'll fix it in
        # another branch dedicated to I/O. Instead of nLeafs tuples,
        # it only goes up to np.where(memBuffer.data == 0)[0][0].
        # Chaning this save affects the load, too, though
        for i, sample in enumerate(memBuffer.data):
            if isinstance(sample, tuple):
                dg['states'][i] = sample[0]
                dg['actions'][i] = sample[1]
                dg['rewards'][i] = sample[2]
                dg['next_states'][i] = sample[3]
                dg['dones'][i] = sample[4]
            else:
                dg['states'][i] = np.zeros(memBuffer.data[0][0].shape)
                dg['actions'][i] = sample
                dg['rewards'][i] = sample
                dg['next_states'][i] = np.zeros(memBuffer.data[0][0].shape)
                dg['dones'][i] = sample
        


#============================================
#             load_deque_memory
#============================================
def load_deque_memory(h5f, maxLen):
    """
    handles loading the agent's memories when the memory buffer is a
    deque.

    parameters:
    -----------
        pass

    raises:
    -------
        pass

    returns:
    --------
        pass
    """
    # Read data from file
    states = h5f['deque/states'][:]
    actions = h5f['deque/actions'][:]
    rewards = h5f['deque/rewards'][:]
    nextStates = h5f['deque/next_states'][:]
    dones = h5f['deque/dones'][:]
    # Package data into the memory buffer
    memBuffer = collections.deque(maxlen=maxLen)
    for experience in zip(states, actions, rewards, nextStates, dones):
        memBuffer.append(experience)
    return memBuffer


#============================================
#            load_sumtree_memory
#============================================
def load_sumtree_memory(h5f, maxLen):
    """
    handles loading the agent's memories when the memory buffer is a
    sum tree.

    parameters:
    -----------
        pass

    raises:
    -------
        pass

    returns:
    --------
        pass
    """
    # Read data from file
    nLeafs, dataPointer = h5f['sumtree/counters'][:]
    tree = h5f['sumtree/tree'][:]
    states = h5f['sumtree/data/states'][:]
    actions = h5f['sumtree/data/actions'][:]
    rewards = h5f['sumtree/data/rewards'][:]
    nextStates = h5f['sumtree/data/next_states'][:]
    dones = h5f['sumtree/data/dones'][:]
    # Package the data into the buffer
    memBuffer = nu.SumTree(nLeafs)
    memBuffer.dataPointer = dataPointer
    memBuffer.tree = tree
    for i, exp in enumerate(zip(states, actions, rewards, nextStates, dones)):
        memBuffer.data[i] = exp
    return memBuffer


#============================================
#               load_memory
#============================================
def load_memory(savePath, memLen):
    """
    Loads in the memory buffer.

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
    # Open file for reading
    with h5py.File(os.path.join(savePath, 'memory_buffer.h5'), 'r') as h5f:
        # Buffer is a deque
        if 'deque' in h5f.keys():
            memBuffer = load_deque_memory(h5f, memLen)
        # Buffer is a SumTree
        elif 'sumtree' in h5f.keys():
            memBuffer = load_sumtree_memory(h5f, memLen)
        else:
            raise KeyError("Error, could not infer type of memory buffer!")
    return memBuffer


#============================================
#             save_train_params
#============================================
def save_train_params(trainParams, savePath):
    """
    This function saves the crucial training parameters needed in order
    to continue where training left off.

    Parameters:
    -----------
        pass

    Raises:
    -------
        pass

    Returns:
    --------
        None
    """
    # Unpack the training parameters
    episode, \
    decayStep, \
    step, \
    fixedQStep, \
    totRewards, \
    epRewards, \
    state, \
    frameStack, \
    memBuffer = trainParams
    # Create hdf5 file
    with h5py.File(os.path.join(savePath, 'training_params.h5'), 'w') as h5f:
        # Save the counters: startEp, decayStep, step, and fixedQStep
        # Compression here won't make much of a difference since, for
        # the games I'm using, these data are all quite small
        counters = np.array([episode, decayStep, step, fixedQStep])
        h5f.create_dataset('counters', data=counters, compression='gzip', compression_opts=4)
        # Total rewards
        h5f.create_dataset('totrewards', data=totRewards, compression='gzip', compression_opts=4)
        # Episode rewards
        h5f.create_dataset('eprewards', data=epRewards, compression='gzip', compression_opts=4)
        # State
        h5f.create_dataset('state', data=state, compression='gzip', compression_opts=4)
    # Memory. This is where compression and io speed is important,
    # because this is huge (or can be)
    save_memory(memBuffer, savePath)
    

#============================================
#             load_train_params
#============================================
def load_train_params(savePath, memLen):
    """
    This function reads in the data saved to the files produced in
    save_train_params so that training can continue where it left off.

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
    # Create hdf5 file
    with h5py.File(os.path.join(savePath, 'training_params.h5'), 'r') as h5f:
        # Load counters
        episode, decayStep, step, fixedQStep = list(h5f['counters'][:])
        # Rewards
        totRewards = list(h5f['totrewards'][:])
        epRewards = list(h5f['eprewards'][:])
        # State (it's actually the stacked set of processed frames)
        state = h5f['state'][:]
        # Unpack the stacked state such that each individual frame
        # in the stack is an element of the deque. The shape of
        # state = (shrinkRows, shrinkCols, stackSize) and I want
        # each element of frameStack to be one of the
        # (shrinkRows, shrinkCols) frames
        frameStack = [state[:,:,i] for i in range(state.shape[-1])]
        frameStack = collections.deque(frameStack, maxlen=state.shape[-1])
    # Memory
    memBuffer = load_memory(savePath, memLen)
    # Package the parameters
    trainParams = (
        episode,
        decayStep,
        step,
        fixedQStep,
        totRewards,
        epRewards,
        state,
        frameStack,
        memBuffer
    )
    return trainParams


#============================================
#                 save_model
#============================================
def save_model(savePath, saveFile, qNet, tNet, trainParams, saveParams):
    """
    Driver function for saving the networks and, if applicable, the
    relevant training parameters for beginning from where we left off.

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
    # Save the primary network
    qNet.model.save(os.path.join(savePath, saveFile + '.h5'))
    # Save the target network, if applicable
    if isinstance(tNet, nw.DQN):
        tNet.model.save(os.path.join(savePath, saveFile + '-target.h5'))
    # Save the training parameters, if applicable
    if saveParams:
        save_train_params(trainParams, savePath)
