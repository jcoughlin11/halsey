"""
Title:   qagent.py
Purpose: Contains the Agent class for using Q-learning techniques
Notes:
"""
import anna
from anna.agents.baseagent import BaseAgent


# ============================================
#                  QAgent
# ============================================
class QAgent(BaseAgent):
    """
    The primary object manager for using Q-learning techniques.

    Attributes:
    -----------
        pass

    Methods:
    --------
        pass
    """

    # -----
    # constructor
    # -----
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
        super().__init__()

    # -----
    # train
    # -----
    def train(self):
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
        # If continuing training, load the checkpoint files
        if self.relay.clArgs.continueTraining:
            pass
        # Otherwise, instantiate new objects
        else:
            navigator = anna.navigation.utils.get_new_navigator(
                self.relay.run.envName,
                self.relay.navigation,
                self.relay.explore,
                self.relay.frame,
            )
            brain = anna.brains.utils.get_new_brain(
                self.relay.network,
                navigator.env.action_space.n,
                navigator.frameManager,
            )
            memory = anna.memory.utils.get_new_memory(self.relay.memory)
            trainer = anna.trainers.utils.get_new_trainer(
                self.relay.training, self.relay.run.timeLimit
            )
            memory.pre_populate(navigator)
            navigator.reset()
        # Training loop
        while not trainer.doneTraining:
            brain, memory, navigator = trainer.train(brain, memory, navigator)
            self.ioManager.writer.save_checkpoint(
                brain, memory, navigator, trainer
            )
        # Save a copy of the parameter file
        self.ioManager.writer.save_param_file(self.relay)
        # Clean up (close env, files, etc)
        navigator.env.close()
        # If early stopping, exit
        if trainer.earlyStop:
            return False
        else:
            return True
