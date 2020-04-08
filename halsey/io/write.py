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

    NOTE: environments can only be deterministically recreated if either the
    deterministic version of the ROM has been used (e.g., SpaceInvadersDeterministic-v4)
    or if the environment is completely written in python (as this allows for
    deep copies of the class to be made). This currently only supports
    native gym environments. For custom envs, need to add a check of some kind
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
            ds = f.create_dataset(attrName, data=attrVal)


# ============================================
#               save_pipeline
# ============================================
def save_pipeline(pipeline):
    """
    Doc string.
    """
    with h5py.File(os.path.join(os.getcwd(), "pipeline.h5"), "w") as f:
        for attrName, attrVal in policy.__dict__.items():
            ds = f.create_dataset(attrName, data=attrVal)


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

    NOTE: The replay buffer needs to be saved in a more efficient manner
    """
    with h5py.File(os.path.join(os.getcwd(), "memory.h5"), "w") as f:
        for attrName, attrVal in policy.__dict__.items():
            ds = f.create_dataset(attrName, data=attrVal)


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
