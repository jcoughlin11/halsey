"""
Title: utils.py
Purpose:
Notes:
    * The memory class stores information (experience) about previously
        visited states
    * The memory class handles constructing samples of data to use in
        learning
"""
from .experience.experiencememory import ExperienceMemory


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
    if memoryParams.mode == "experience":
        memory = ExperienceMemory(memoryParams)
    return memory
