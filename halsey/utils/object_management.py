"""
Title: object_management.py
Purpose: Handles creation of new objects.
Notes:
    * The trainer class oversees the navigator, brain, and memory
    * The navigator oversees the frame manager, action chooser, and env
    * The brain oversees the network(s) and the learning method
    * The memory oversees the experience buffer and sampling method
    * The trainer class contains the main training loop
"""
from .env import build_env
from .validation import optionRegister


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
    frameManager = optionRegister[folio.frame.mode](folio.frame)
    actionManager = optionRegister[folio.action.mode](folio.action)
    env = build_env(folio.run.envName)
    navigator = optionRegister[folio.navigation.mode](
        env, frameManager, actionManager
    )
    brain = optionRegister[folio.brain.mode](folio.brain)
    memory = optionRegister[folio.memory.mode](folio.memory)
    trainer = optionRegister[folio.training.mode](
        folio.training, navigator, brain, memory
    )
    # Pre-populate the memory buffer
    trainer.pre_populate()
    return trainer
