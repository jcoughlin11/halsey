"""
Title: write.py

Notes:
"""
import os

import h5py
import numpy as np


# ============================================
#              save_checkpoint
# ============================================
def save_checkpoint(instructor):
    """
    Doc string.
    """
    save_navigator(instructor.navigator)
    save_memory(instructor.memory)
    save_brain(instructor.brain)
    save_training_params(instructor.trainParams)


# ============================================
#               save_navigator
# ============================================
def save_navigator(navigator):
    """
    Doc string.
    """
    save_env(navigator.env)
    save_policy(navigator.policy)
    save_pipeline(navigator.pipeline)
    save_state(navigator.state)
    if navigator.navParams is not None:
        save_nav_params(navigator.navParams)


# ============================================
#                 save_env
# ============================================
def save_env(env):
    """
    Doc string.

    NOTE: environments can only be deterministically recreated if
    either the deterministic version of the ROM has been used (e.g.,
    SpaceInvadersDeterministic-v4) or if the environment is completely
    written in python (as this allows for deep copies of the class to
    be made). This currently only supports native gym environments. For
    custom envs, need to add a check of some kind.
    """
    envState = env.unwrapped.clone_full_state()
    np.save("envState", envState)


# ============================================
#               save_policy
# ============================================
def save_policy(policy):
    """
    Doc string.
    """
    with h5py.File(os.path.join(os.getcwd(), "policy.h5"), "w") as f:
        for attrName, attrVal in policy.__dict__.items():
            f.create_dataset(attrName, data=attrVal)


# ============================================
#               save_pipeline
# ============================================
def save_pipeline(pipeline):
    """
    Doc string.
    """
    with h5py.File(os.path.join(os.getcwd(), "pipeline.h5"), "w") as f:
        for attrName, attrVal in pipeline.__dict__.items():
            f.create_dataset(attrName, data=attrVal)


# ============================================
#                save_state
# ============================================
def save_state(state):
    """
    Doc string.
    """
    np.save(os.path.join(os.getcwd(), "current_state"), state)


# ============================================
#             save_nav_params
# ============================================
def save_nav_params(navParams):
    """
    Doc string.
    """
    raise NotImplementedError


# ============================================
#                save_memory
# ============================================
def save_memory(memory):
    """
    Doc string.
    """
    with h5py.File(os.path.join(os.getcwd(), "memory.h5"), "w") as f:
        for attrName, attrVal in memory.__dict__.items():
            if attrName != "replayBuffer":
                f.create_dataset(attrName, data=attrVal)
    save_replay_buffer(memory.replayBuffer)


# ============================================
#            save_replay_buffer
# ============================================
def save_replay_buffer(replayBuffer):
    """
    Doc string.

    NOTE: Use array.size to get number of elements in array and
    array.itemsize to get the size (in bytes) of each array element.
    The product will give the size of the array in bytes and this can
    be used to break the buffer up into several smaller files of a
    given fixed size. This should be the method used when doing IO in
    serial. For parallel, just use one file per worker.
    """
    # Create the empty datasets to store the memory components in
    # a group named after the memory type. This makes reading the
    # data back in easier because type(replayBuffer) can be identified
    # without passing any flags
    nSamples = len(replayBuffer)
    stateShape = [nSamples] + list(replayBuffer[0][0].shape)
    with h5py.File(os.path.join(os.getcwd(), "replay_buffer.h5"), "w") as h5f:
        g = h5f.create_group("ExperienceMemory")
        g.create_dataset(
            "states",
            shape=stateShape,
            compression="gzip",
            compression_opts=4,
            dtype=np.float,
        )
        g.create_dataset(
            "actions",
            (nSamples,),
            compression="gzip",
            compression_opts=4,
            dtype=np.int,
        )
        g.create_dataset(
            "rewards",
            (nSamples,),
            compression="gzip",
            compression_opts=4,
            dtype=np.float,
        )
        g.create_dataset(
            "next_states",
            shape=stateShape,
            compression="gzip",
            compression_opts=4,
            dtype=np.float,
        )
        g.create_dataset(
            "dones",
            (nSamples,),
            compression="gzip",
            compression_opts=4,
            dtype=np.int,
        )
        # Loop over each sample in the buffer
        for i, sample in enumerate(replayBuffer):
            g["states"][i] = sample[0]
            g["actions"][i] = sample[1]
            g["rewards"][i] = sample[2]
            g["next_states"][i] = sample[3]
            g["dones"][i] = sample[4]


# ============================================
#                save_brain
# ============================================
def save_brain(brain):
    """
    Doc string.
    """
    raise NotImplementedError


# ============================================
#            save_training_params
# ============================================
def save_training_params(trainingParams):
    """
    Doc string.
    """
    raise NotImplementedError
