"""
Title: utils.py
Purpose:
Notes:
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
    if trainParams.mode == "qtrainer":
        trainer = QTrainer()
    return trainer
