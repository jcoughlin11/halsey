"""
Title: setup.py
Notes:
"""
import gin
import gym
from rich.traceback import install as install_rich_traceback

from halsey.io.logging import setup_loggers
from halsey.io.read import parse_cl_args
from halsey.manager.manager import Manager

from .endrun import endrun
from .misc import create_output_directory
from .misc import get_data_format
from .misc import lock_file


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
    manager = Manager()
    create_output_directory(manager.outputDir)
    lock_file(clArgs.paramFile, manager.outputDir)
    return manager


# ============================================
#                 get_trainer
# ============================================
@gin.configurable("training")
def get_trainer(trainerCls, params):
    """
    Doc string.
    """
    channelsFirst = get_data_format()
    env = get_env()
    memory = get_memory_object()
    explorer = get_explorer()
    pipeline = get_pipeline(channelsFirst)
    inShape = pipeline.inputShape
    nActions = env.action_space.n
    brain = get_brain(inShape, nActions, channelsFirst)
    return trainerCls(env, pipeline, explorer, memory, brain, params)


# ============================================
#                  get_tester
# ============================================
def get_tester():
    """
    Doc string.
    """
    raise NotImplementedError


# ============================================
#                 get_analyst
# ============================================
def get_analyst():
    """
    Doc string.
    """
    raise NotImplementedError


# ============================================
#                   get_env
# ============================================
@gin.configurable("game")
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
#                get_pipeline
# ============================================
@gin.configurable("pipeline")
def get_pipeline(channelsFirst, pipelineCls, params):
    """
    Doc string.
    """
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
#                 get_brain
# ============================================
@gin.configurable("learning")
def get_brain(inShape, nActions, channelsFirst, brainCls, params):
    """
    Doc string.
    """
    nets = get_networks(inShape, nActions, channelsFirst)
    return brainCls(nets, params)


# ============================================
#                get_networks
# ============================================
@gin.configurable("networks")
def get_networks(inShape, nActions, channelsFirst, nets, params):
    """
    Doc string.
    """
    s = inShape
    n = nActions
    c = channelsFirst
    return [nn(s, n, c, p) for (nn, p) in zip(nets, params)]


# ============================================
#                get_explorer
# ============================================
@gin.configurable("explorer")
def get_explorer(explorerCls, params):
    """
    Doc string.
    """
    return explorerCls(params)
