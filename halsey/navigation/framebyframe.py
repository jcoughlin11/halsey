"""
Title: framebyframe.py
Purpose: Class for stepping through the game world one frame at a time.
Notes:
"""
import halsey

from halsey.utils.experience import Experience

from .basenavigator import BaseNavigator


# ============================================
#          FrameByFrameNavigator
# ============================================
@halsey.utils.validation.register_navigator
class FrameByFrameNavigator(BaseNavigator):
    """
    Represents a fully Markovian process whereby the agent takes in and
    processes each and every frame of the game.

    Attributes
    ----------
    See halsey.navigation.BaseNavigator

    Methods
    -------
    transition(brain=None, mode="train")
        For a given state, chooses an action based on mode and then
        observes the results of having taken that action. The current
        state is then updated to the resulting next state (frame) of
        the game.
    """

    # -----
    # constructor
    # -----
    def __init__(self, env, navParams, frameManager, actionManager):
        """
        Sets up the navigator object.

        Parameters
        ----------
        env : gym.Env
            The interface between the game and the agent.

        navParams : halsey.utils.folio.Folio
            An object containing the navigation-specific data read in
            from the parameter file.

        frameManager : halsey.frames.FrameManager
            The image-processing pipeline used on incoming game states.

        actionManager : halsey.actions.ActionManager
            Handles selecting an action for the current state.

        Raises
        ------
        None

        Returns
        -------
        None
        """
        super().__init__(env, navParams, frameManager, actionManager)

    # -----
    # transition
    # -----
    def transition(self, brain=None, mode="train"):
        """
        For the current state, this method handles selecting an action,
        taking that action, observing the results of that action, and
        then updating the current state to the resulting state.

        Parameters
        ----------
        brain : halsey.brains.QBrain
            This is only used if mode=test or if mode=train (and in
            that case, it's only used if the exploration-exploitation
            strategy deems it necessary).

        mode : str
            One of either train, test, or random. Determines how the
            action is to be selected. If train is selected, then the
            chosen exploration-exploitation strategy is employed. If
            test is selected, then the agent's current knowledge
            (network) is always used. If random is selected, then the
            action is determined randomly, with each action having an
            equal chance of being selected.

        Raises
        ------
        None

        Returns
        -------
        experience : halsey.utils.experience.Experience
            The state, action, reward, next state, done information
            packaged into a container object.
        """
        # Choose an action
        action = self.actionManager.choose(self.state, self.env, brain, mode)
        # Take the action
        nextState, reward, done, _ = self.env.step(action)
        # Process the next state
        nextState = self.frameManager.process_frame(nextState)
        # Package up the experience
        experience = Experience(self.state, action, reward, nextState, done)
        return experience
