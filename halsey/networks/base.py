"""
Title: base.py
Notes:
"""
import tensorflow as tf


# ============================================
#                BaseNetwork
# ============================================
class BaseNetwork(tf.keras.Model):
    """
    Doc string.
    """

    # -----
    # constructor
    # -----
    def __init__(self, params):
        """
        Doc string.
        """
        super().__init__()
        self.params = params
