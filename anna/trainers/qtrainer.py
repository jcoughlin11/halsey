"""
Title:   qtrainer.py
Purpose: Contains the QTrainer class.
Notes:
"""


#============================================
#                 QTrainer
#============================================
class QTrainer:
    """
    Handles choosing actions, interacting with the environment, and
    calling the learn methods.

    Attributes:
    -----------
        pass

    Methods:
    --------
        pass
    """
    #-----
    # constructor
    #-----
    def __init__(self):
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
        self.done = False
        self.saveCheckpoint = False
        self.saveFinal = False
        self.params = self.initialize_params()
        self.frameHandler = FrameHandler()
        self.actionSelector = anna.navigation.get_action_selector()
