"""
Title: setup.py
Notes:
"""
import gin
import gym
import numpy as np
import tensorflow as tf

from halsey.losses.custom import lossRegister
from halsey.optimizers.custom import optimizerRegister

from .endrun import endrun


# ============================================
#              setup_instructor
# ============================================
@gin.configurable("training")
def setup_instructor(
    envName,
    channelsFirst,
    instructorCls,
    brainCls,
    policyCls,
    pipelineCls,
    memoryCls,
    navCls,
    nets,
):
    """
    Primary driver function for instantiating a new instructor object.

    The instructor object contains the desired training loop. As such,
    this function also handles instantiation of those additional
    objects that are required for training.

    Parameters
    ----------
    envName : str
        The name of the gym environment to serve as the interface
        between the game and halsey.

    channelsFirst : bool
        If True, the batchsize is the first dimension of the shape
        of the input to the network(s). Otherwise, it's the last.

    instructorCls : :py:class:`halsey.instructors.base.BaseInstructor`
        The object containing the desired training loop.

    brainCls : :py:class:`halsey.brains.base.BaseBrain`
        The object containing the learning method. Also stores the
        neural networks.

    policyCls : :py:class:`halsey.policies.base.BasePolicy`
        The object containing the method used to select an action to
        take during each training step.

    pipelineCls : :py:class:`halsey.pipelines.base.BasePipeline`
        The object responsible for converting the frames returned by
        gym into the format required by the neural network(s).

    memoryCls : :py:class:`halsey.memory.base.BaseMemory`
        The object responsible for storing and managing the
        interactions between halsey and the game. Stores the pipeline.

    navCls : :py:class:`halsey.navigation.base.BaseNavigator`
        The object responsible for interacting with the environment.
        Stores the gym environment and policy.

    nets : list
        List of neural networks to employ. Each network should be an
        instance of :py:class:`halsey.networks.base.BaseNetwork`.

    Raises
    ------
    Void

    Returns
    -------
    instructor : `halsey.instructors.base.BaseInstructor`
        The instantiated instructor object ready to being training.

    """
    env = build_env(envName)
    nActs = env.action_space.n
    policy = policyCls()
    pipeline = pipelineCls(channelsFirst)
    inShape = get_input_shape(channelsFirst, pipeline)
    navigator = navCls(env, policy, pipeline)
    memory = memoryCls()
    networks = [netCls(channelsFirst, inShape, nActs) for netCls in nets]
    brain = brainCls(networks)
    instructor = instructorCls(navigator, memory, brain)
    return instructor


# ============================================
#                 build_env
# ============================================
def build_env(envName):
    """
    Instantiates the gym environment that serves as the interface
    between halsey and the game.

    Parameters
    ----------
    envName : str
        The name of the gym environment to serve as the interface
        between the game and halsey.

    Raises
    ------
    gym.error.UnregisteredEnv
        Raised when an unknown environment name is encountered.

    gym.error.DeprecatedEnv
        Raised when an old version of a known environment is
        encountered.
    """
    try:
        env = gym.make(envName)
    except gym.error.UnregisteredEnv:
        msg = f"Unknown environment `{envName}`."
        endrun(msg)
    except gym.error.DeprecatedEnv:
        msg = f"Using deprecated version of environment `{envName}`."
        endrun(msg)
    return env


# ============================================
#                prep_sample
# ============================================
def prep_sample(sample):
    """
    Batches all aspects of the sample.

    Parameters
    ----------
    sample : np.ndarray
        Array of shape (batchSize, 5). Each row is an experience:
        state, action, reward, next state, done.

    Raises
    ------
    pass

    Returns
    -------
    tuple
        An experience tuple, except that every element is now a batched
        numpy array
    """
    states = np.stack(sample[:, 0]).astype(np.float)
    actions = sample[:, 1].astype(np.int)
    rewards = sample[:, 2].astype(np.float)
    nextStates = np.stack(sample[:, 3]).astype(np.float)
    dones = sample[:, 4].astype(np.bool)
    return (states, actions, rewards, nextStates, dones)


# ============================================
#              get_input_shape
# ============================================
def get_input_shape(channelsFirst, pipeline):
    """
    Doc string.
    """
    if channelsFirst:
        inShape = (pipeline.traceLen, pipeline.cropHeight, pipeline.cropWidth)
    else:
        inShape = (pipeline.cropHeight, pipeline.cropWidth, pipeline.traceLen)
    return inShape


# ============================================
#               get_loss_func
# ============================================
def get_loss_func(lossName):
    """
    Creates the loss function object based on the string
    representation.

    Custom loss functions must not share a name with those losses
    already registered with tensorflow.

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
    try:
        loss = tf.keras.losses.get(lossName)
    except ValueError:
        if lossName in lossRegister:
            loss = lossRegister[lossName]
        else:
            msg = f"Unrecognized loss function `{lossName}`."
            endrun(msg)
    return loss


# ============================================
#                get_optimizer
# ============================================
def get_optimizer(optimizerName, learningRate):
    """
    Creates optimizer object from its name.

    Parameters
    ----------
    optimizerName : str
        The name of the optimizer to use.

    learningRate : float
        The step size to use during back propagation.

    Raises
    ------
    None

    Returns
    -------
    optimizer : tf.keras.optimizers.Optimizer
        The actual optimizer object to perform minimization of the loss
        function.
    """
    try:
        optimizer = tf.keras.optimizers.get(optimizerName)
    except ValueError:
        if optimizerName in optimizerRegister:
            optimizer = optimizerRegister[optimizerName]
        else:
            msg = f"Unrecognized optimizer `{optimizerName}`."
            endrun(msg)
    optimizer.learning_rate = learningRate
    return optimizer
