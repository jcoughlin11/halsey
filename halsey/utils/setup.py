"""
Title:      setup.py
Purpose:    Contains functions for initializing a run.
Notes:
"""
import gin

from halsey.io.logger import setup_loggers
from halsey.io.read import parse_cl_args

from .endrun import endrun


# ============================================
#                   setup
# ============================================
def setup():
    """
    Primary driver function for initializing a run.

    Handles parsing the command-line arguments, setting up the loggers,
    reading in the gin configuration file,  and then instantiating an
    agent.

    Parameters
    ----------
    Void

    Raises
    ------
    IOError
        If the gin configuration file cannot be read.

    Returns
    -------
    agent : halsey.agents.base.BaseAgent
        The object doing the learning, being tested, and/or being
        analyzed.
    """
    clArgs = parse_cl_args()
    setup_loggers(clArgs.silent, clArgs.noColoredLogs)
    try:
        gin.parse_config_file(clArgs.paramFile)
    except IOError as e:
        msg = f"Could not read config file: `{clArgs.paramFile}`"
        endrun(e, msg)
    except ValueError as e:
        msg = f"Unknown configurable or parameter in `{clArgs.paramFile}`."
        endrun(e, msg)
    agent = get_agent()
    return agent


# ============================================
#                  get_agent
# ============================================
@gin.configurable("run")
def get_agent(agentCls, policyCls, navCls, memoryCls, modelCls, trainParams):
    """
    Oversees the instantiation of the chosen agent class.

    Specifying an agent means specifying the training loop.

    In order to train, the agent needs: a replay buffer (memory
    object), a navigator, and a model.

    The replay buffer stores the feedback provided by the environment
    after performing an action (experience).

    The navigator is the object that manages how actions are selected
    during training (the policy), how the game states are traversed
    (e.g., one frame at a time, skip frames), and processes the feedback
    from the environment.

    The model contains the neural network(s) and the learning method
    (e.g., DQL, DDQL).

    These objects are created here and then passed on to the agent's
    constructor. This is not done in setup() because you cannot pass
    configurable parameters to the function where the gin config file
    is parsed.

    Parameters
    ----------
    agentCls : :py:class:`halsey.agents.base.BaseAgent`
        Specifies the training loop to use.

    policyCls : :py:class:`halsey.policies.base.BasePolicy`
        Specifies which action selection policy to use during training.

    navCls : :py:class:`halsey.navigation.base.BaseNavigator`
        Specifies which navigation object to use during training.

    memoryCls : :py:class:`halsey.memory.base.BaseMemory`
        Specifies the type of replay buffer to use.

    modelCls : :py:class:`halsey.models.base.BaseModel`
        Specifies the learning method to use.

    trainParams : dict
        Dictionary containing all of the relevant training
        hyperparameters.

    Raises
    ------
    pass

    Returns
    -------
    agent : :py:class:`halsey.agents.base.BaseAgent`
        The instantiated agent object.
    """
    trainPolicy = policyCls()
    navigator = navCls(policy=trainPolicy)
    memory = memoryCls()
    model = modelCls()
    agent = agentCls(model, memory, navigator, trainParams)
    return agent
