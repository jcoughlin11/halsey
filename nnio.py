"""
Title: io.py
Author: Jared Coughlin
Date: 7/30/19
Purpose: Contains tools related to reading and writing files to disk
Notes:
"""
import os

import numpy


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
        "architecture"
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
    if hyperparams["architecture"] not in archRegister:
        raise ValueError("Error, unrecognized network architecture!")
    # Double DQN requires fixed-Q
    if params["double_dqn"] and not params["fixed_Q"]:
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
        raise IOError
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
                print("Hyperparameter not found!")
                raise IOError
            hyperparams[key] = value
    # Check for option conflicts
    check_agent_option_conflicts(hyperparams)
    return hyperparams


#============================================
#             save_train_params
#============================================
def save_train_params(decay, rewards, mem, path, qstep):
    """
    This function saves the crucial training parameters needed in order
    to continue where training left off.

    Parameters:
    -----------
        ep : int
            The most recent episode to have finished.

        decay : int
            The value of the decay_step used in explore-exploit epsilon
            greedy.

        rewards : list
            List of the total reward earned for each completed episode.

        mem : deque
            The memory buffer for the current training session.

        path : string
            Place to save this information.

        qstep : int
            The current step that we're on with regards to when the
            targetQNet should be updated. Only matters if using fixed-Q.

    Raises:
    -------
        pass

    Returns:
    --------
        None
    """
    # Episode, decay, and episode rewards
    with open(os.path.join(path, "ep_decay_reward.txt"), "w") as f:
        f.write(str(decay) + "\n")
        f.write(str(qstep) + "\n")
        for i in range(len(rewards)):
            f.write(str(rewards[i]) + "\n")
    # States
    states = np.array([s[0] for s in mem], ndmin=3)
    np.savez(os.path.join(path, "exp_states"), *states)
    # Actions
    actions = np.array([s[1] for s in mem])
    np.savez(os.path.join(path, "exp_actions"), *actions)
    # Rewards
    exp_rewards = np.array([s[2] for s in mem])
    np.savez(os.path.join(path, "exp_rewards"), *exp_rewards)
    # Next states
    next_states = np.array([s[3] for s in mem], ndmin=3)
    np.savez(os.path.join(path, "exp_next_states"), *next_states)
    # Dones
    dones = np.array([s[4] for s in mem])
    np.savez(os.path.join(path, "exp_dones"), *dones)


#============================================
#             load_train_params
#============================================
def load_train_params(path, max_len):
    """
    This function reads in the data saved to the files produced in
    save_train_params so that training can continue where it left off.

    Parameters:
    -----------
        path : string
            The path to the required data files.

        max_len : int
            The maximum length of the memory buffer.

    Raises:
    -------
        pass

    Returns:
    --------
        train_params : tuple
            (start_episode, decay_step, totalRewards, memory, qstep).
    """
    # Read the ep_decay_reward file
    with open(os.path.join(path, "ep_decay_reward.txt"), "r") as f:
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
    states = np.load(os.path.join(path, "exp_states.npz"))
    actions = np.load(os.path.join(path, "exp_actions.npz"))
    rewards = np.load(os.path.join(path, "exp_rewards.npz"))
    next_states = np.load(os.path.join(path, "exp_next_states.npz"))
    dones = np.load(os.path.join(path, "exp_dones.npz"))
    # Sanity check
    nstates = len(states.files)
    if (
        len(actions.files) != nstates
        or len(rewards.files) != nstates
        or len(next_states.files) != nstates
        or len(dones.files) != nstates
    ):
        print(
            "Error, length of read in states array does not match "
            "length of actions, rewards, next_states, or dones!"
        )
        sys.exit()
    # Get experience tuples to fill mem buffer (state, action, reward,
    # next_state, done)
    buf = collections.deque(maxlen=max_len)
    for i in range(nstates):
        key = "arr_" + str(i)
        exp = (
            states[key],
            actions[key],
            rewards[key],
            next_states[key],
            dones[key],
        )
        buf.append(exp)
    # Package everything up
    train_params = (ep, decay_step, ep_rewards, buf, qstep)
    return train_params
