"""
Title: write.py

Notes:
"""
import os
import shutil
import stat

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz
import yaml

from halsey.utils.setup import prep_sample


# ============================================
#           lock_parameter_file
# ============================================
def lock_parameter_file(paramFile):
    """
    Doc string.
    """
    paramLockFile = os.path.join(os.getcwd(), "params.lock")
    shutil.copyfile(paramFile, paramLockFile)
    os.chmod(paramLockFile, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


# ============================================
#               save_object_state
# ============================================
def save_object_state(obj, name):
    """
    Doc string.
    """
    objState = obj.get_state()
    fname = os.path.join(os.getcwd(), name + "_state.yaml")
    with open(fname, "w") as f:
        yaml.safe_dump(objState, f)


# ============================================
#              save_checkpoint
# ============================================
def save_checkpoint(instructor):
    """
    Doc string.
    """
    # Save instructor's state (e.g., current episode, etc.)
    save_object_state(instructor, "instructor")
    # Save brain
    save_brain(instructor.brain, instructor.checkpointManager)
    # Save the environment-specific variables
    save_navigator(instructor.navigator)
    # Save the memory
    save_memory(instructor.memory)


# ============================================
#                save_brain
# ============================================
def save_brain(brain, checkpointManager):
    """
    Doc string.
    """
    save_object_state(brain, "brain")
    checkpointManager.save()


# ============================================
#              save_navigator
# ============================================
def save_navigator(navigator):
    """
    Doc string.
    """
    save_environment(navigator.env)
    save_object_state(navigator.policy, "policy")


# ============================================
#              save_environment
# ============================================
def save_environment(env):
    """
    Doc string.

    NOTE: This really only works with deterministic environments.
    """
    envState = env.unwrapped.clone_full_state()
    np.save("envState", envState)


# ============================================
#                save_memory
# ============================================
def save_memory(memory):
    """
    Doc string.
    """
    save_object_state(memory, "memory")
    save_replay_buffer(memory.replayBuffer)


# ============================================
#             save_replay_buffer
# ============================================
def save_replay_buffer(replayBuffer):
    """
    The rewards and dones arrays are sparse, and so can be saved as
    such, which helps reduce file size. Next, for each episode, you
    only really need to save the states, not the nextStates. This
    is because state i's nextState is just state i+1's state.
    What if the last entry in the buffer is sampled, though? If this
    is a continued training run and the nextStates haven't been saved,
    then you don't have access to the nextState. Could always just take
    the saved action, in that case.
    """
    replayBuffer = prep_sample(np.array(replayBuffer))
    states, actions, rewards, nextStates, dones = replayBuffer
    save_array(rewards, "rewards", sparse=True)
    save_array(dones, "dones", sparse=True)
    save_array(actions, "actions")
    save_array(states, "states")


# ============================================
#                 save_array
# ============================================
def save_array(array, name, sparse=False):
    """
    Doc string.
    """
    fname = os.path.join(os.getcwd(), name)
    if sparse:
        array = csr_matrix(array)
        save_npz(fname, array)
    else:
        np.save(fname, array)
