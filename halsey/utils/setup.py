"""
Title: setup.py
Notes:
"""
import gin
import gym
from rich.traceback import install as install_rich_traceback

from halsey.io.logging import setup_loggers
from halsey.io.read import parse_cl_args

from .endrun import endrun
from .misc import get_data_format


# ============================================
#                    setup
# ============================================
def setup():
    """
    Doc string.
    """
    install_rich_traceback()
    clArgs = parse_cl_args()
    setup_loggers()
    try:
        gin.parse_config_file(clArgs.paramFile)
    except IOError:
        msg = f"Could not read config file: `{clArgs.paramFile}`"
        endrun(msg)
    except ValueError:
        msg = f"Unknown configurable or parameter in `{clArgs.paramFile}`."
        endrun(msg)
    return clArgs


# ============================================
#                 get_trainer
# ============================================
@gin.configurable("training")
def get_trainer(trainerCls, params):
    """
    Doc string.
    """
    game = get_game()
    memory = get_memory_object()
    inputShape = game.pipeline.inputShape
    nActions = game.env.action_space.n
    channelsFirst = game.pipeline.channelsFirst
    nets = get_networks(inputShape, nActions, channelsFirst)
    brain = get_brain(nets)
    return trainerCls(game, memory, brain, params)


# ============================================
#                  get_game
# ============================================
@gin.configurable("game")
def get_game(gameCls, gameName, params):
    """
    Doc string.
    """
    env = get_env(gameName)
    explorer = get_explorer()
    pipeline = get_pipeline()
    return gameCls(env, explorer, pipeline, params)


# ============================================
#                   get_env
# ============================================
def get_env(gameName):
    """
    Doc string.
    """
    try:
        env = gym.make(gameName)
    except gym.error.UnregisteredEnv:
        msg = f"Unknown environment `{envName}`."
        endrun(msg)
    except gym.error.DeprecatedEnv:
        msg = f"Using deprecated version of environment `{envName}`."
        endrun(msg)
    return env


# ============================================
#                get_explorer
# ============================================
@gin.configurable("explorer")
def get_explorer(explorerCls, params):
    """
    Doc string.
    """
    return explorerCls(params)


# ============================================
#                get_pipeline
# ============================================
@gin.configurable("pipeline")
def get_pipeline(pipelineCls, params):
    """
    Doc string.
    """
    channelsFirst = get_data_format()
    return pipelineCls(channelsFirst, params)


# ============================================
#              get_memory_object
# ============================================
@gin.configurable("memory")
def get_memory_object(memoryCls, params):
    """
    Doc string.
    """
    return memoryCls(params)


# ============================================
#                get_networks
# ============================================
@gin.configurable("networks")
def get_networks(inputShape, nActions, channelsFirst, nets, params):
    """
    Doc string.
    """
    s = inputShape
    n = nActions
    c = channelsFirst
    return [nn(s, n, c, p) for (nn, p) in zip(nets, params)]


# ============================================
#                 get_brain
# ============================================
@gin.configurable("learning")
def get_brain(nets, brainCls, params):
    """
    Doc string.
    """
    return brainCls(nets, params)
