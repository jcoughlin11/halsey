"""
Title: validation.py
Purpose:
Notes:
"""
import os

from .folio import Folio


# ============================================
#                 Registers
# ============================================
# Brain parameters
convRegister = [
    "conv1",
]

duelingRegister = [
    "dueling1",
]

rnnRegister = [
    "rnn1",
]

registers = Folio()
setattr(registers, "convNetRegister", convRegister)
setattr(registers, "duelingNetRegister", duelingRegister)
setattr(registers, "rnnRegister", rnnRegister)


# ============================================
#             validate_params
# ============================================
def validate_params(folio):
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
    pass


# ============================================
#                is_empty_dir
# ============================================
def is_empty_dir(directory):
    """
    Checks to see if the output directory tree is empty.

    This is to prevent overwriting a directory already in use by
    another run.

    Parameters
    ----------
    directory : str
        The root of the output directory tree.

    Raises
    ------
    None

    Returns
    -------
    isEmpty : bool
        If True then the directory tree is empty and we can proceed.
    """
    isEmpty = True
    # Walk the directory tree
    for root, dirs, files in os.walk(directory):
        # If there are files, then we're done
        if len(files) != 0:
            isEmpty = False
            break
    return isEmpty
