"""
Title: utils.py
Purpose:
Notes:
"""
from . import dueling1


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
    if arch == "dueling1":
        net = dueling1.build_dueling1_net(inputShape, channelsFirst, nActions)
    return net
