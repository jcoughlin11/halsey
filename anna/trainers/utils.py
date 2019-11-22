"""
Title: utils.py
Purpose: Contains methods related to setting up a new trainer object.
Notes:
"""
from anna.trainers.qtrainer import QTrainer


# ============================================
#              get_new_trainer
# ============================================
def get_new_trainer(trainParams, timeLimit):
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
    if trainParams.mode == "qtrainer":
        trainer = QTrainer(trainParams, timeLimit)
    return trainer
