"""
Title: utils.py
Purpose:
Notes:
    * The trainer class oversees the navigator, brain, and memory
    * The trainer class contains the main training loop
"""


# ============================================
#              get_new_trainer
# ============================================
def get_new_trainer():
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
    # Create a new navigator
    navigator = anna.navigation.utils.get_new_navigator()
    # Create a new brain
    brain = anna.brains.utils.get_new_brain()
    # Create a new memory object
    memory = anna.memory.utils.get_new_memory()
    if trainParams.mode == "qtrainer":
        trainer = QTrainer()
    return trainer
