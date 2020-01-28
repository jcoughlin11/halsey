"""
Title: env.py
Purpose: Creates and manages the game environment.
Notes:
    * The environment handles interaction between the game and the
        Agent
"""
import gym

import halsey


# ============================================
#                 build_env
# ============================================
def build_env(envName):
    """
    Creates a new game environment.

    Parameters
    ----------
    envName : str
        The name of the game environment to build.

    Raises
    ------
    None

    Returns
    -------
    env : gym.Env
        The fully constructed game environment.
    """
    env = gym.make(envName)
    return env


# ============================================
#                 get_shapes
# ============================================
def get_shapes(arch, frameParams, envName):
    """
    Sets up the shape of the input to the neural network(s) and gets
    the size of the game's action space, which is used in determining
    the neural network's output size.

    The actual input shape can be in one of two forms: NCHW or NHWC,
    where N is the batch size, C is the number of channels (the trace
    length), H is the number of rows in the image, and W is the number
    of columns.

    Which form is used depends on the type of neural network being used
    as well as which device the code is being run on. See
    :py:meth:`~halsey.utils.gpu.set_channels` for more.

    Parameters
    ----------
    arch : str
        The name of the neural network architecture being used.

    frameParams : halsey.utils.folio.Folio
        A container for the parameters in the frame section of the
        parameter file.

    envName : str
        The name of the gym environment being used.

    Raises
    ------
    None

    Returns
    -------
    None
    """
    # Determine whether or not the number of channels go first
    channelsFirst = halsey.utils.gpu.set_channels(arch)
    # Set the input shape accordingly
    if channelsFirst:
        inputShape = [
            frameParams.traceLen,
            frameParams.shrinkRows,
            frameParams.shrinkCols,
        ]
    else:
        inputShape = [
            frameParams.shrinkRows,
            frameParams.shrinkCols,
            frameParams.traceLen,
        ]
    # Now get the size of the environment's action space
    env = build_env(envName)
    # Reset must be called before the environment can be used at all
    nActions = env.reset()
    nActions = env.action_space.n
    env.close()
    return inputShape, nActions, channelsFirst
