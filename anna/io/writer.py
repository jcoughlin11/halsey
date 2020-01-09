"""
Title: writer.py
Purpose:
Notes:
"""
import os
import stat

import tensorflow as tf
import yaml


# ============================================
#                   Writer
# ============================================
class Writer:
    """
    Doc string.

    Attributes:
    -----------
        pass

    Methods:
    --------
        pass
    """

    # -----
    # save_params
    # -----
    def save_params(self, params, outputDir):
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
        # Set up the lock file
        paramLockFile = os.path.join(outputDir, "params.lock")
        # Save the params to a read-only file (writing will fail if a
        # lock file already exists because it will be read-only)
        try:
            with open(paramLockFile, "w") as f:
                yaml.dump(params, f)
        except PermissionError:
            raise
        # Change permissions to read-only
        os.chmod(paramLockFile, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

    # -----
    # save_brain
    # -----
    def save_models(self, brain, outputDir):
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
        for attr, attrValue in brain.__dict__.items():
            # Save the networks
            if isinstance(attrValue, tf.keras.Model):
                fname = os.path.join(outputDir, attr + ".h5")
                attrValue.save(fname)
