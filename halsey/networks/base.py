"""
Title: base.py
Notes:
"""
import gin
import tensorflow as tf


# ============================================
#                BaseNetwork
# ============================================
@gin.configurable("network")
class BaseNetwork(tf.keras.Model):
    """
    Doc string.

    Attributes
    ----------
    pass

    Methods
    -------
    pass
    """

    # -----
    # constructor
    # -----
    def __init__(self, netParams=None):
        """
        Doc string.
        """
        pass
