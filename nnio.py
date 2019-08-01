"""
Title: io.py
Author: Jared Coughlin
Date: 7/30/19
Purpose: Contains tools related to reading and writing files to disk
Notes:
"""
import collections
import os

import h5py
import numpy as np

import nnetworks as nw
import nnutils as nu


# Architecture register
archRegister = ["conv1",
    "dueling1",
    "perdueling1",
    "rnn1"
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
        "restart_training",
        "shrink_cols",
        "shrink_rows",
        "save_period",
        "test_flag",
        "trace_len",
        "train_flag",
    ]
    string_params = [
        "architecture",
        "ckpt_file",
        "env_name",
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
    # Double DQN requires fixed-Q
    if params["enable_double_dqn"] and not params["enable_fixed_Q"]:
        raise ValueError("Error, double dqn requires the use of fixed Q!")


# ============================================
#             read_hyperparams
# ============================================
def read_hyperparams(fname):
    """
    Reads in the parameter file that contains the network's
    hyperparameters. The layout is:

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
    max_episode_steps   : int, max number of steps per episode
    memory_size         : int, max number of experiences to store in
                          memory buffer
    n_episodes          : int, number of episodes to train for
    n_stacked_frames    : int, number of frames to stack
    per_a               : float, alpha parameter in eq. 1 of Schaul16
    per_b               : float, beta param in IS weights of Schaul16
    per_b_anneal        : float, annealment rate of IS weights
    per_e               : float, epsilon parameter in prop. prior. of
                          Schaul16
    pretrain_len        : int, num experiences to initially fill mem
    pretrain_max_ep_len : int, max ep length when filling mem buffer
    render_flag         : int, if 1, render scene during testing
    restart_training    : int, if 1 start from beginning, if 0, cont
    save_path           ; string, path of checkpoint and param file
    save_period         : int, save model every savePeriod episodes
    shrink_cols         : int, y size of shrunk frame
    shrink_rows         : int, x size of shrunk frame
    test_flag           : int, if 1, test the agent
    trace_len           : int, num connected frames in RNN sample
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
    # Assume the file is in the current working directory
    fname = os.path.join(os.getcwd(), fname)
    # Make sure the file exists
    if not os.path.isfile(fname):
        raise FileNotFoundError
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
    # Check for option conflicts
    check_agent_option_conflicts(hyperparams)
    return hyperparams


#============================================
#                save_memory
#============================================
def save_memory(memBuffer, savepath):
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
    # Create hdf5 file
    h5f = h5py.File('memory_buffer.h5', 'w')
    # Case 1: memBuffer is a deque
    if isinstance(memBuffer, collections.deque):
        h5f.create_dataset('deque', data=np.array(memBuffer))
    # Case 2: memBuffer is a SumTree. In this case, the whole data
    # structure needs to be saved
    elif isinstance(memBuffer, nu.SumTree):
        # Counters
        h5f.create_dataset('counters',
            data=np.array([memBuffer.nLeafs, memBuffer.dataPointer])
        )
        h5f.create_dataset('tree', data=memBuffer.tree)
        h5f.create_dataset('data', data=memBuffer.data)
    # Unrecognized case
    else:
        h5f.close()
        raise TypeError("Error, unrecognized memory buffer type!")
    # Close file
    h5f.close()


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
    h5f = h5py.File(os.path.join(savePath, 'memory_buffer.h5'), 'r')
    # Buffer is a deque
    if 'deque' in h5f.keys():
        memBuffer = collections.deque(h5f['deque'][:], maxlen=memLen)
    # Buffer is a SumTree
    elif 'tree' in h5f.keys():
        # Read and set the counters
        nLeafs, dataPointer = list(h5f['counters'][:])
        memBuffer = nu.SumTree(nLeafs)
        memBuffer.dataPointer = dataPointer
        # Read the tree and experience data
        memBuffer.tree = h5f['tree'][:]
        memBuffer.data = h5f['data'][:]
    else:
        h5f.close()
        raise KeyError("Error, could not infer type of memory buffer!")
    # Close file
    h5f.close()
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
    h5f = h5py.File(os.path.join(savePath, 'training_params.h5'), 'w')
    # Save the counters: startEp, decayStep, step, and fixedQStep
    counters = np.array([episode, decayStep, step, fixedQStep])
    h5f.create_dataset('counters', data=counters)
    # Total rewards
    h5f.create_dataset('totrewards', data=totRewards)
    # Episode rewards
    h5f.create_dataset('eprewards', data=epRewards)
    # State
    h5f.create_dataset('state', data=state)
    h5f.close()
    # Memory
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
    h5f = h5py.File(os.path.join(savePath, 'training_params.h5'), 'r')
    # Load counters
    episode, decayStep, step, fixedQStep = list(h5f['counters'][:])
    # Rewards
    totRewards = h5f['totrewards'][:]
    epRewards = h5f['eprewards'][:]
    # State and frame stack
    state = h5f['state'][:]
    frameStack = collections.deque(state, maxlen=state.shape[-1])
    # Close file
    h5f.close()
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
