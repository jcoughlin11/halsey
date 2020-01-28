"""
Title: utils.py
Purpose: Creates a new trainer object.
Notes:
    * The trainer class oversees the navigator, brain, and memory
    * The trainer class contains the main training loop
"""
import halsey

from .qtrainer import QTrainer


# ============================================
#              get_new_trainer
# ============================================
def get_new_trainer(folio):
    """
    Handles the creation of a new trainer object.

    Parameters
    ----------
    folio : halsey.utils.folio.Folio
        The relevant data read in from the parameter file in object
        form.

    Raises
    ------
    None

    Returns
    -------
    trainer : halsey.trainers.Trainer
        The manager of the training loop.
    """
    # Create a new navigator
    navigator = halsey.navigation.utils.get_new_navigator(
        folio.navigation, folio.action, folio.frame, folio.run.envName
    )
    # Create a new brain
    brain = halsey.brains.utils.get_new_brain(
        folio.brain,
        navigator.env.action_space.n,
        navigator.frameManager.inputShape,
        navigator.frameManager.channelsFirst,
    )
    # Create a new memory object
    memory = halsey.memory.utils.get_new_memory(folio.memory)
    if folio.training.mode == "qtrainer":
        trainer = QTrainer(folio.training, navigator, brain, memory)
    # Pre-populate the memory buffer
    trainer.pre_populate()
    return trainer
