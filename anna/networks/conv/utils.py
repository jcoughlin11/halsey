"""
Title: utils.py
Purpose:
Notes:
"""
from . import conv1


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
    if arch == "conv1":
        net = conv1.build_conv1_net(inputShape, channelsFirst, nActions)
    return net
