"""
Title: setup.py
Notes:
"""
import gin

from halsey.io.read import parse_cl_args

from .register import registry


# ============================================
#                  initialize
# ============================================
def initialize():
    """
    Handles the necessary groundwork for a `halsey` run.

    Information about the run comes from two places: the parameter file
    and the command line. This function oversees the parsing of that
    information, the preservation of that information, and the setup
    for the loggers.
    """
    clArgs = parse_cl_args()
    gin.parse_config_file(clArgs.paramFile)
    return clArgs


# ============================================
#              setup_instructor
# ============================================
@gin.configurable("training")
def setup_instructor(instructorCls, trainParams):
    """
    Instantiates the `instructor` object based on the values given in
    the parameter file.
    """
    brain = setup_brain()
    navigator = setup_navigator()
    instructor = registry[instructorCls](brain, navigator, trainParams)
    return instructor


# ============================================
#                setup_proctor
# ============================================
def setup_proctor():
    """
    Instantiates the `proctor` object based on the values given in
    the parameter file.
    """
    raise NotImplementedError

# ============================================
#                setup_analyst
# ============================================
def setup_analyst():
    """
    Instantiates the `analyst` object based on the values given in
    the parameter file.
    """
    raise NotImplementedError


# ============================================
#                setup_brain
# ============================================
@gin.configurable("learning")
def setup_brain(brainCls, learningParams):
    """
    Instantiates the `brain` object based on the values given in the
    parameter file.
    """
    memory = setup_memory()
    networks = setup_networks()
    brain = registry[brainCls](memory, networks, learningParams)
    return brain


# ============================================
#                setup_memory
# ============================================
@gin.configurable("memory")
def setup_memory(memoryCls, memoryParams):
    """
    Instantiates the `memory` object based on the values given in the
    parameter file.
    """
    memory = registry[memoryCls](memoryParams)
    return memory


# ============================================
#               setup_networks
# ============================================
@gin.configurable("networks")
def setup_networks(netClasses, netParams):
    """
    Instantiates the `network` object(s) based on the values given in
    the parameter file.
    """
    optimizers = setup_optimizers()
    losses = setup_losses()
    networks = []
    for (nn, opt, lf, params) in zip(netClasses, optimizers, losses, netParams):
        networks.append(registry[nn](opt, lf, params))
    return networks


# ============================================
#               setup_optimizers
# ============================================
@gin.configurable("optimizers")
def setup_optimizers(optimizers, optimizerParams):
    """
    Instantiates the optimizer classes. One for each network.
    """
    opts = []
    for (opt, params) in zip(optimizers, optimizerParams):
        optimizer = tf.keras.optimizers.get(opt)
        for key, value in params:
            if hasattr(optimizer, key):
                setattr(optimizer, key, value)
        opts.append(optimizer)
    return opts
        

# ============================================
#                 setup_losses
# ============================================
@gin.configurable("losses")
def setup_losses(lossFunctions, lossParams):
    """
    Sets the desired loss function, one for each network.

    It seems like each loss in keras is implemented as both a
    function and a class. Looking at the tf source code, it's weird,
    though, the way the attributes are set, so I'm going to keep
    working with only the function versions. Some loss functions
    (such as mse) don't take any key word args while others (such as
    categorical cross entropy) do, so I have to allow for their
    existence.
    """
    losses = []
    for (lf, params) in zip(lossFunctions, lossParams):
        losses.append(HalseyLoss(lf, params))
    return losses


# ============================================
#              setup_navigator
# ============================================
@gin.configurable("navigation")
def setup_navigator(navigatorCls, navigatorParams):
    """
    Instantiates the `navigator` object based on the values given in
    the parameter file.
    """
    env = setup_environment()
    explorer = setup_explorer()
    imagePipeline = setup_image_pipeline()
    navigator = registry[navigatorCls](env, explorer, imagePipeline, navigatorParams)
    return navigator


# ============================================
#             setup_environment
# ============================================
@gin.configurable("environment")
def setup_environment(envName):
    """
    Instantiates the `gym` environment for the run.
    """
    env = gym.make(envName)
    return env


# ============================================
#               setup_explorer
# ============================================
@gin.configurable("exploration")
def setup_explorer(explorerCls, explorerParams):
    """
    Instantiates the `explorer` object based on the values given in the
    parameter file.
    """
    explorer = registry[explorerCls](explorerParams)
    return explorer


# ============================================
#           setup_image_pipeline
# ============================================
@gin.configurable("images")
def setup_image_pipeline(imagePipelineCls, imagePipelineParams):
    """
    Instantiates the `pipeline` object based on the values given in
    the parameter file.
    """
    imagePipeline = registry[imagePipelineCls](imagePipelineParams)
    return imagePipeline
