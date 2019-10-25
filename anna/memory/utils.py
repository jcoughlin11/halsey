"""
Title: utils.py
Purpose: Contains functions related to creating a new memory object.
Notes:
"""
from anna.memory.episode_memory import EpisodeMemory
from anna.memory.experience_memory import ExperienceMemory


# ============================================
#              get_new_memory
# ============================================
def get_new_memory(memoryParams):
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
    if memoryParams.mode == "experience":
        memory = ExperienceMemory(memoryParams)
    elif memoryParams.mode == "episode":
        memory = EpisodeMemory(memoryParams)
    return memory
