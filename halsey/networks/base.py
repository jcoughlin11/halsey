"""
Title: base.py
Notes:
"""
import tensorflow as tf

from halsey.utils.endrun import endrun


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
    def __init__(self, channelsFirst, params):
        """
        Doc string.
        """
        super().__init__()
        if channelsFirst:
            self.dataFormat = "channels_first"
        else:
            self.dataFormat = "channels_last"
        self.optimizerName = params["optimizer"]
        self.lossName = params["loss"]
        self.learningRate = params["learningRate"]
        try:
            loss = tf.keras.losses.get(self.lossName)
        except ValueError:
            msg = f"Unrecognized loss function `{self.lossName}`."
            endrun(msg)
        try:
            optimizer = tf.keras.optimizers.get(self.optimizerName)
        except ValueError:
            msg = f"Unrecognized optimizer `{self.optimizerName}`."
            endrun(msg)
        optimizer.learning_rate = self.learningRate
        self.lossFunction = loss
        self.optimizer = optimizer
