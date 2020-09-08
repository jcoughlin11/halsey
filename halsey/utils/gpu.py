"""
Title: gpu.py
Notes:
"""
import tensorflow as tf


# ============================================
#                  using_gpu
# ============================================
def using_gpu():
    """
    Determines whether or not to use a gpu.
    """
    usingGPU = False
    nGpu = len(tf.config.list_physical_devices("GPU"))
    if tf.test.is_built_with_gpu_support() and nGpu > 0:
        usingGPU = True
    return usingGPU


# ============================================
#               get_data_format
# ============================================
def get_data_format(netType):
    """
    Determines whether or not nChannels is the first part of the
    shape or the last (not counting batchSize, which is always the
    first dimension when passing data to the network).

    RNNs require NCHW. For CNN architectures, though, the shape
    depends on the device doing the training. If it's a CPU we need
    NHWC since CPUs don't support NCHW for CNNs (because the former
    is more efficient). If it's a GPU, though, NCHW is more
    efficient.
    """
    dataFormat = "channels_first"
    if netType != "recurrent" and not using_gpu():
        dataFormat = "channels_last"
    return dataFormat


# ============================================
#               get_input_shape
# ============================================
def get_input_shape(dataFormat, trace, height, width):
    """
    Returns either CHW or HWC depending on the value of dataFormat.
    The batch dimension is excluded.
    """
    inputShape = None
    if dataFormat == "channels_first":
        inputShape = [trace, height, width]
    elif dataFormat == "channels_last":
        inputShape = [height, width, trace]
    return inputShape
