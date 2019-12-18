"""
Title: utils.py
Purpose:
Notes:
"""
from . import rnn1


# ============================================
#                   build_net
# ============================================
def build_net(arch, inputShape, channelsFirst, nActions):
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
    if arch == "rnn1":
        net = rnn1.build_rnn1_net(inputShape, channelsFirst, nActions)
    return net
