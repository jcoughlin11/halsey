"""
Title: basenavigator.py
Purpose: Contains the BaseNavigator class
Notes:
"""


# ============================================
#               BaseNavigator
# ============================================
class BaseNavigator:
    """
    The navigator represents how the agent interacts with the game
    environment.

    This interaction includes receiving and processing frames of the
    game state from the game, deciding which action to take in a
    given situation, as well as the agent's **receptiveness**.

    Receptiveness determines how fine-grained the agent's attention is.
    In other words, does the agent perceive every frame of the game, or
    only certain frames (Markovian vs. non-Markovian). What details of
    these frames does the agent pay attention to and ultimately decide
    to save? That sort of thing.

    Attributes
    ----------
    actionManager : anna.actions.ActionManager
        The object responsible for how the agent chooses which action
        to take in a given state.

    env : gym.Env
        The interface between the agent and the game.

    frameManager : anna.frames.FrameManager
        The image processing pipeline for the frames taken in by the
        agent.

    state : np.ndarray
        The current game state.

    Methods
    -------
    reset()
        Resets the game back to its original state.
    """

    # -----
    # constructor
    # -----
    def __init__(self, navParams, env, frameManager, actionManager):
        """
        Initializes the object.

        Parameters
        ----------
        navParams : anna.utils.folio.Folio
            Object containing the navigation-specific parameters from
            the parameter file.

        env : gym.Env
            The interface between the game and the agnet.

        frameManager : anna.frames.FrameManager
            The image processing pipeline for the frames taken in by
            the agent.

        actionManager : anna.actions.ActionManager
            The object responsible for how the agent chooses which
            action to take in a given state.

        Raises
        ------
        None

        Returns
        -------
        None
        """
        self.env = env
        self.frameManager = frameManager
        self.actionManager = actionManager
        self.state = None

    # -----
    # reset
    # -----
    def reset(self):
        """
        Reverts the game back to its original state.

        Parameters
        ----------
        None

        Raises
        ------
        None

        Returns
        -------
        None
        """
        # Reset the game environment
        state = self.env.reset()
        # Process the initial frame
        self.state = self.frameManager.process_frame(state, newEpisode=True)
