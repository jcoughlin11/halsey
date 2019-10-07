"""
Title: utils.py
Purpose: Contains functions related to creating a new memory object.
Notes:
"""


#============================================
#              get_new_memory
#============================================
def get_new_memory(memoryParams, batchSize, navigator):
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
    # Build the memory object of the desired type
    if memoryParams.mode == 'experience':
        memory = ExperienceMemory(memoryParams, batchSize)
    elif memoryParams.mode == 'episode':
        memory = EpisodeMemory(memoryParams, batchSize)
    # Pre-populate the memory buffer to avoid the empty-memory problem
    memory.pre_populate(navigator.env, navigator.frameHandler)
    return memory
