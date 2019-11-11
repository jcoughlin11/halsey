"""
Title:   writer.py
Purpose: Contains the writer class
Notes:
"""
import os
import pickle
from collections import deque

import h5py
import numpy as np
import tensorflow as tf
import yaml

from anna.memory.experience_memory import ExperienceMemory
from anna.utils.relay import Relay, class_to_dict


# ============================================
#                  Writer
# ============================================
class Writer:
    """
    Used to write data to files.

    Attributes:
    -----------
        pass

    Methods:
    --------
        pass
    """

    # -----
    # constructor
    # -----
    def __init__(self):
        """
        Doc string.

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
        self.baseDir = None
        self.fileBase = None

    # -----
    # set_params
    # -----
    def set_params(self, params):
        """
        Doc string.

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
        self.baseDir = params.outputDir
        self.fileBase = params.fileBase

    # -----
    # save_checkpoint
    # -----
    def save_checkpoint(self, brain, memory, navigator, trainer):
        """
        Doc string.

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
        # Make the checkpoints directory, if needed
        dirName = os.path.join(self.baseDir, "checkpoints")
        if not os.path.isdir(dirName):
            os.makedirs(dirName)
        # Save the brain
        self.save_brain(dirName, brain)
        # Save the navigator
        self.save_navigator(dirName, navigator)
        # Save the trainer
        self.save_object(dirName, trainer, "trainer")
        # Save the memory
        self.save_memory(dirName, memory)

    # -----
    # save_brain
    # -----
    def save_brain(self, dirName, brain):
        """
        Doc string.

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
        # Extract everything except the networks from the brain
        partialBrain = Relay()
        networks = {}
        for k, v in brain.__dict__.items():
            if isinstance(v, tf.keras.Model):
                networks.update({k: v})
            else:
                setattr(partialBrain, k, v)
        # Serialize the partial brain object
        self.save_object(dirName, partialBrain, "partial_brain")
        # Save the networks
        for k, v in networks:
            fn = os.path.join(dirName, k + ".h5")
            v.save(fn)

    # -----
    # save_navigator
    # -----
    def save_navigator(self, dirName, navigator):
        """
        Doc string.

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
        # Save the environment
        self.save_env(dirName, navigator.env)
        # Save the frame manager
        self.save_object(dirName, navigator.frameManager, "frame_manager")
        # Save the action manager
        self.save_object(dirName, navigator.actionManager, "action_manager")
        # Save the state
        self.save_state(dirName, navigator.state)

    # -----
    # save_env
    # -----
    def save_env(self, dirName, env):
        """
        Doc string.

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
        pass

    # -----
    # save_state
    # -----
    def save_state(self, dirName, state):
        """
        Doc string.

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
        fn = os.path.join(dirName, "state.npy")
        np.save(fn, state)

    # -----
    # save_object
    # -----
    def save_object(self, dirName, obj, baseName):
        """
        Doc string.

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
        fn = os.path.join(dirName, baseName + ".pickle")
        with open(fn, "wb") as f:
            pickle.dump(obj, f)

    # -----
    # save_memory
    # -----
    def save_memory(self, dirName, memory):
        """
        Doc string.

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
        # Serialize everything except the buffer
        partialMem = Relay()
        for k, v in memory.__dict__.items():
            if not isinstance(k, deque):
                setattr(partialMem, k, v)
        self.save_object(dirName, partialMem, "partial_mem")
        # Save the memory buffer
        if isinstance(memory, ExperienceMemory):
            self.save_experience_memory_buffer(dirName, memory.buffer)

    # -----
    # save_memory_buffer
    # -----
    def save_experience_memory_buffer(self, dirName, buf):
        """
        Doc string. The memory buffer will be a deque of Experiences.
        Each experience is comprised of a: state, action, reward,
        next state, and done. The actions, rewards, and dones will all
        have shape (1, buffer.maxlen) since there's only one number
        per experience. Those can be packaged together. The states
        and next states are where this eats up HDD space. Also, you
        can't slice a deque.

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
        with h5py.File(os.path.join(dirName, "memory_buffer.h5"), "w") as h5f:
            # Create the empty datasets to store the memory components in
            # a group named after the memory type. This makes reading the
            # data back in easier because type(memBuffer) can be identified
            # without passing any flags
            nSamples = len(buf)
            stateShape = list(nSamples) + list(buf[0].state.shape)
            g = h5f.create_group("experience_memory")
            g.create_dataset(
                "states",
                shape=stateShape,
                compression="gzip",
                compression_opts=4,
                dtype=np.float,
            )
            g.create_dataset(
                "actions",
                shape=(nSamples,),
                compression="gzip",
                compression_opts=4,
                dtype=np.int,
            )
            g.create_dataset(
                "rewards",
                shape=(nSamples,),
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
                shape=(nSamples,),
                compression="gzip",
                compression_opts=4,
                dtype=np.int,
            )
            # Loop over each sample in the buffer
            for i, exp in enumerate(buf):
                g["states"][i] = exp.state
                g["actions"][i] = exp.action
                g["rewards"][i] = exp.reward
                g["next_states"][i] = exp.nextState
                g["dones"][i] = exp.done

    # -----
    # save_param_file
    # -----
    def save_param_file(self, relay):
        """
        Doc string.

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
        fn = os.path.join(self.baseDir, self.fileBase + "_params_backup.yaml")
        # If the file already exists, don't overwrite it
        if not os.path.isfile(fn):
            relay = class_to_dict(relay, {})
            with open(fn, "w") as f:
                yaml.dump(f, relay, default_flow_style=False)
