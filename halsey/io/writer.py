"""
Title: writer.py
Purpose: Handles the saving of all files.
Notes:
"""
import os
import stat
import sys

import tensorflow as tf
import yaml


# ============================================
#                   Writer
# ============================================
class Writer:
    """
    Handles the saving of all files.

    Attributes
    ----------
    None

    Methods
    -------
    save_models(brain, outputDir)
        Saves the network(s). This includes the weights, architecture,
        metrics, and optimizer state.

    save_params(params, outputDir)
        Creates a copy of the given parameter file.
    """

    # -----
    # save_params
    # -----
    def save_params(self, params, outputDir):
        """
        Saves a copy of the given parameter file.

        The resulting file is read-only and has a lock extension to
        indicate this. The purpose of this file is both to provide
        reference about the run to the user in the future, as well as
        to allow easier resuming of training without having to worry
        about potential changes made to the parameter file in the
        interim.

        Parameters
        ----------
        params : dict
            The parameters read from the given parameter file.

        outputDir : str
            The name of the base output directory where the lock file
            will be saved.

        Raises
        ------
        None

        Returns
        -------
        None
        """
        # Set up the lock file
        paramLockFile = os.path.join(outputDir, "params.lock")
        # Save the params to a read-only file (writing will fail if a
        # lock file already exists because it will be read-only)
        try:
            with open(paramLockFile, "w") as f:
                yaml.dump(params, f)
        except PermissionError:
            msg = "Error: Lock file `{}.lock` already exists.".format(
                paramLockFile
            )
            print(msg)
            sys.exit(1)
        # Change permissions to read-only
        os.chmod(paramLockFile, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

    # -----
    # save_brain
    # -----
    def save_models(self, brain, outputDir):
        """
        Saves the brain's network(s).

        This includes the network architecture, weights, metrics, and
        optimizer state.

        Parameters
        ----------
        brain : halsey.brains.QBrain
            The object containing the network(s) to be saved.

        outputDir : str
            The full path to the brain output directory where the files
            will be saved.

        Raises
        ------
        None

        Returns
        -------
        None
        """
        for attr, attrValue in brain.__dict__.items():
            # Save the networks
            if isinstance(attrValue, tf.keras.Model):
                fname = os.path.join(outputDir, attr + ".h5")
                try:
                    attrValue.save(fname)
                except (OSError, IOError):
                    msg = "Error: Can't save model `{}` to file `{}`.".format(
                        attr, fname
                    )
                    print(msg)
                    sys.exit(1)
