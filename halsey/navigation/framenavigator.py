"""
Title:      framenavigator.py
Purpose:    Class for stepping through the game world one frame at a
                time.
Notes:
"""
import gin

from .base import BaseNavigator


# ============================================
#               FrameNavigator
# ============================================
@gin.configurable(blacklist=["policy"])
class FrameNavigator(BaseNavigator):
    """
    Represents a fully Markovian process whereby the agent takes in and
    processes each and every frame of the game.

    Attributes
    ----------
    See :py:class:`halsey.navigation.base.BaseNavigator`

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
    def __init__(self, env, policy=None):
        """
        Sets up the navigator object.

        Parameters
        ----------
        env : gym.Env
            The interface between the game and the agent.

        policy : :py:class:`halsey.policies.BasePolicy`
            Handles selecting an action for the current state.

        Raises
        ------
        None

        Returns
        -------
        None
        """
        super().__init__(env, policy)

    # -----
    # transition
    # -----
    def transition(self, model=None, mode="train"):
        """
        For the current state, this method handles selecting an action,
        taking that action, observing the results of that action, and
        then updating the current state to the resulting state.

        Parameters
        ----------
        model : :py:class:`halsey.models.base.BaseModel`
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
        experience : tuple
            The state, action, reward, next state, and done information
        """
        # Choose an action
        action = self.policy.choose(self.state, self.env, model, mode)
        # Take the action
        nextState, reward, done, _ = self.env.step(action)
        # Package up the experience
        experience = (self.state, action, reward, nextState, done)
        return experience
