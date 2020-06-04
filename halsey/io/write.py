"""
Title: write.py
Notes:
"""


# ============================================
#                save_model
# ============================================
def save_model(manager):
    """
    Doc string.

    Things that need to be saved (in order to continue training later on):
        * Game
            * emulator (env) state
            * most recently "completed" frameStack
                * This can be obtained from the memory buffer
            * explorer parameters
            * pipeline parameters
                * These can probably be obtained from the parameter file, but
                    it's possible that pipelines with other parameters get
                    introduced in the future. I don't think I'm going to worry
                    about that now, though
            * any class-specific parameters that might arise
        * Memory
            * memory-specific parameters
            * replay buffer
        * brain
            * brain-specific parameters
            * networks
                * optimizers
                * loss functions
                * number of networks
                * architectures
                    * Everything except for the optimizer state and network
                        weights can be obtained from the parameter file
                        (use a checkpointmanager from tf)
        * training parameters

    The real question is how this should all be saved? What is an efficient
    way to do this?

    The real issue is the memory buffer.

    Certain parameters can be obtained from the locked version of the parmater
    file, so that's easy.

    It might make sense to save other parameters in a yaml file, since they're
    generally just counters and such? It's possible in the future that isn't
    the case, though. Maybe h5? Use awkward for ragged arrays?

    The gym env is saved as a numpy array, so it can be saved in npz or h5
    formats.

    Structure of output directory:
    |-output/
        |-param.lock
        |-checkpoint_1/
        |-...
        |-checkpoint_N/
            |-tf checkpoint manager file(s)
            |-object-specific parameter file(s)
            |-game env
            |-replay buffer file(s)

    Saving multiple copies of the full replay buffer can get expensive, so
    maybe just save the diff? That is, save the full buffer for the first
    checkpoint and then, for each subsequent checkpoint, just save the new
    experiences
    """
    raise NotImplementedError
