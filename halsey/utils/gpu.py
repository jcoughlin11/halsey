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
