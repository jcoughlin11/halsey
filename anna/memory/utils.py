"""
Title: utils.py
Purpose: Contains functions related to creating a new memory object.
Notes:
"""


#============================================
#              get_new_memory
#============================================
def get_new_memory(memoryParams, batchSize):
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
    if memoryParams.mode == 'experience':
        memory = ExperienceMemory(memoryParams, batchSize)
    elif memoryParams.mode == 'episode':
        memory = EpisodeMemory(memoryParams, batchSize)
    return memory
