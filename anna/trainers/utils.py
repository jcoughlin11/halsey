"""
Title: utils.py
Purpose:
Notes:
    * The trainer class oversees the navigator, brain, and memory
    * The trainer class contains the main training loop
"""
import anna


# ============================================
#              get_new_trainer
# ============================================
def get_new_trainer(folio):
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
    navigator = anna.navigation.utils.get_new_navigator(
        folio.navigation, folio.action, folio.frame, folio.run.envName
    )
    # Create a new brain
    brain = anna.brains.utils.get_new_brain(
        folio.brain,
        navigator.env.action_space.n,
        navigator.frameManager.inputShape,
        navigator.frameManager.channelsFirst,
    )
    # Create a new memory object
    memory = anna.memory.utils.get_new_memory(folio.memory)
    if folio.training.mode == "qtrainer":
        trainer = anna.trainers.qtrainer.QTrainer(
            folio.training, navigator, brain, memory
        )
    return trainer
