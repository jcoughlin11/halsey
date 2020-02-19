"""
Title:      object_management.py
Purpose:    Contains functions related to creating the various objects
                used throughout halsey.
Notes:
"""
import gin


# ============================================
#                  get_agent
# ============================================
@gin.configurable("agent")
def get_agent(agentType, trainFlag):
    """
    Oversees the instantiation of the chosen agent class.

    Specifying an agent means specifying the learning method (e.g.,
    specifying @QAgent for agent.agentType in the .gin file means that
    you are choosing to use the vanilla deep-q method presented in
    [Mnih13]_).

    In order to train, the agent also needs: a neural network, a replay
    buffer (memory object), and a data processing pipeline. These
    objects are created here and then passed on to the agent.

    Parameters
    ----------
    agentType : :py:class:`halsey.agents.base.BaseAgent`
        Specifies the learning method to use (e.g., DQL, DDQL).

    trainFlag : bool
        If True, train the agent. If False, skip training.

    Raises
    ------
    pass

    Returns
    -------
    agent : :py:class:`halsey.agents.base.BaseAgent`
        The instantiated agent object.

    References
    ----------
    .. [Mnih13] Minh, V., **et al**., "Playing Atari with Deep
        Reinforcement Learning," CoRR, vol. 1312, 2013.
    """
    pipeline = get_pipeline()
    memory = get_memory()
    model = get_model()
    agent = agentType(model, memory, pipeline, trainFlag)
    return agent


# ============================================
#               get_pipeline
# ============================================
@gin.configurable("pipeline")
def get_pipeline():
    """
    Handles construction of the data processing pipeline.

    The pipeline represents the preprocessing and sampling procedures
    to perform on the frames returned by gym in order to get them into
    the form expected by the neural network.

    Parameters
    ----------
    pass

    Raises
    ------
    pass

    Returns
    -------
    pass
    """
    pass


# ============================================
#                  get_model
# ============================================
@gin.configurable("model")
def get_model():
    """
    Handles construction of the primary neural network.

    Parameters
    ----------
    pass

    Raises
    ------
    pass

    Returns
    -------
    pass
    """
    pass


# ============================================
#                  get_memory
# ============================================
@gin.configurable("memory")
def get_memory():
    """
    Handles construction of the replay buffer (memory object).

    Parameters
    ----------
    pass

    Raises
    ------
    pass

    Returns
    -------
    pass
    """
    pass
