"""
Title: utils.py
Purpose: Contains functions used in building the desired network.
Notes:
"""


# ============================================
#               build_network
# ============================================
def build_network(
    arch, inputShape, nActions, optimizerName, lossName, learningRate
):
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
    # Set up network architecture
    if arch == "conv1":
        net = build_conv1_net(inputShape, nActions)
    elif arch == "dueling1":
        net = build_dueling1_net(inputShape, nActions)
    elif arch == "rnn1":
        net = build_rnn1_net(inputShape, nActions)
    # Set the optimizer and loss functions appropriately
    optimizer = set_optimizer(optimizerName, learningRate)
    loss = set_loss(lossName)
    # Compile model
    net.compile(optimizer=optimizer, loss=loss)
    return net


# ============================================
#             set_optimizer
# ============================================
def set_optimizer(optimizerName, learningRate):
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
    if optimizerName == "adam":
        optimizer = tf.keras.optimizers.Adam(lr=learningRate)
    return optimizer


# ============================================
#                set_loss
# ============================================
def set_loss(lossName):
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
    if lossName == "mse":
        loss = tf.keras.losses.MeanSquaredError()
    return loss
