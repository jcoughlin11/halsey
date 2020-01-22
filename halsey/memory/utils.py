"""
Title: utils.py
Purpose: Handles the creation of a new memory object
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
    Handles the creation of a new memory object.

    Parameters
    ----------
    memoryParams : halsey.utils.folio.Folio
        Object containing the memory-specific parameters from the
        parameter file.

    Raises
    ------
    None

    Returns
    -------
    None
    """
    if memoryParams.mode == "experience":
        memory = ExperienceMemory(memoryParams)
    return memory
