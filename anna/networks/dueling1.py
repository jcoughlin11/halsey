"""
Title: dueling1.py
Purpose: Contains the dueling1 network architecture.
Notes:
"""
import tensorflow as tf
import tensorflow.keras.backend as K


# ============================================
#            build_dueling1_net
# ============================================
def build_dueling1_net(inputShape, nActions):
    """
        Uses the keras functional API to build the dueling DQN given in
        Wang et al. 2016.

        The basic idea is that calculating a Q value provides an
        estimate how how good a certain action is in a given state.
        However, it provides no information whatsoever on whether or not
        it is desirable to be in that state in the first place.

        Dueling DQN solves this by separating Q = V + A, where V is the
        value stream and it estimates how desirable being in the
        current state is. A is the advantage stream, and it estimates
        the quality of each action for the state.

        Also see:
        https://keras.io/getting-started/functional-api-guide/

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
    # Input layer
    inputLayer = tf.keras.layers.Input(shape=inputShape, name="input")
    # First convolutional layer
    conv1Layer = tf.keras.layers.Conv2D(
        data_format="channels_first",
        filters=16,
        kernel_size=[8, 8],
        strides=[4, 4],
        activation="relu",
        name="conv1",
    )(inputLayer)
    # Second convolutional layer
    conv2Layer = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=[4, 4],
        strides=[2, 2],
        activation="relu",
        name="conv2",
    )(conv1Layer)
    # Flatten layer
    flattenLayer = tf.keras.layers.Flatten()(conv2Layer)
    # Value stream
    valueFC1 = tf.keras.layers.Dense(
        units=512, activation="relu", name="value_fc1"
    )(flattenLayer)
    value = tf.keras.layers.Dense(units=1, activation=None, name="value")(
        valueFC1
    )
    # Advantage stream
    advantageFC1 = tf.keras.layers.Dense(
        units=512, activation="relu", name="activation_fc1"
    )(flattenLayer)
    advantage = tf.keras.layers.Dense(
        units=nActions, activation=None, name="advantage"
    )(advantageFC1)
    # Aggregation layer: (eq. 9 in paper)
    # Q(s,a) = V(s) + [A(s,a) - 1/|A| * sum_{a'} A(s,a')]
    # Using tf ops here results in an error when saving:
    # https://tinyurl.com/y5hyn8zh
    Q = tf.keras.layers.Lambda(
        lambda q: q[0] + (q[1] - K.mean(q[1], axis=1, keepdims=True))
    )([value, advantage])
    # Set the model
    model = tf.keras.models.Model(inputs=inputLayer, outputs=Q)
    return model
