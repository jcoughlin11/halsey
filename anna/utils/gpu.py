"""
Title: gpu.py
Purpose:
Notes:
"""
import tensorflow as tf

from .validation import rnnRegister


# ============================================
#               set_channels
# ============================================
def set_channels(arch):
    """
    Determines if the shape of the input data to the network is NCHW or
    NHWC.

    N is the batch size (number of samples), C is the number of
    channels (the trace length), H is the image height (number of
    rows), and W is the image width (number of columns).

    RNNs require NCHW. For CNN architectures, though, the shape depends
    on the device doing the training. If it's a CPU we need NHWC since
    CPUs don't support NCHW for CNNs (because the former is more
    efficient). If it's a GPU, though, NCHW is more efficient.

    As such, this function decides whether or not the channels should
    be first or last by checking two things: is the network being used
    an RNN? If not, then are we training on a GPU or not? If not, then
    the channels need to be last.

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
    channelsFirst = True
    if arch not in rnnRegister and not using_gpu():
        channelsFirst = False
    return channelsFirst


# ============================================
#                using_gpu
# ============================================
def using_gpu():
    """
    Determines whether or not training is going to happen on a GPU or
    a CPU.

    A GPU can only be used if a.) a GPU is present and b.) tf was built
    with GPU support. In any other case, the CPU must be used.

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
    usingGPU = False
    nGpu = len(tf.config.list_physical_devices("GPU"))
    if tf.test.is_built_with_gpu_support() and nGpu > 0:
        usingGPU = True
    return usingGPU
