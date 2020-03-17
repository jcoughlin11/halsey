"""
Title: base.py
Notes:
"""
import gin
import tensorflow as tf


# ============================================
#                BaseNetwork
# ============================================
@gin.configurable
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
        super().__init__()
