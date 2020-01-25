"""
Title: validation.py
Purpose:
Notes:
"""
import os


# ============================================
#                registers
# ============================================

# -----
# paramRegister
# -----
paramRegister = {
    "envName": str,
    "train": bool,
    "test": bool,
    "timeLimit": int,
    "outputDir": str,
    "fileBase": str,
    "nEpisodes": int,
    "maxEpisodeSteps": int,
    "batchSize": int,
    "savePeriod": int,
    "architecture": str,
    "discount": float,
    "learningRate": float,
    "loss": str,
    "optimizer": str,
    "fixedQSteps": int,
    "maxSize": int,
    "pretrainLen": int,
    "epsDecayRate": float,
    "epsilonStart": float,
    "epsilonStop": float,
    "cropBot": int,
    "cropLeft": int,
    "cropRight": int,
    "cropTop": int,
    "shrinkRows": int,
    "shrinkRows": int,
    "traceLen": int,
}

# -----
# trainerRegister
# -----
trainerRegister = {}

# -----
# brainRegister
# -----
brainRegister = {}

# -----
# networkRegister
# -----
networkRegister = {}
convNetRegister = {}

# -----
# memoryRegister
# -----
memoryRegister = {}
experienceMemoryRegister = {}

# -----
# actionManagerRegister
# -----
actionManagerRegister = {}

# -----
# navigationRegister
# -----
navigatorRegister = {}

# -----
# frameManagerRegister
# -----
frameManagerRegister = {}


# ============================================
#             register_trainer
# ============================================
def register_trainer(cls):
    """
    This wrapper takes in a trainer object and adds it to the training
    register for ease of validation and object creation.

    Parameters
    ----------
    cls : :py:class`~halsey.trainers.basetrainer.BaseTrainer`
        An instance of the BaseTrainer class (or one of its children).

    Raises
    ------
    None

    Returns
    -------
    cls : :py:class`~halsey.trainers.basetrainer.BaseTrainer`
        The unaltered passed-in object.
    """
    trainerRegister[cls.__name__] = cls
    return cls


# ============================================
#              register_brain
# ============================================
def register_brain(cls):
    """
    This wrapper takes in a brain object and adds it to the brain
    register for ease of validation and object creation.

    Parameters
    ----------
    cls : :py:class`~halsey.brains.basebrain.BaseBrain`
        An instance of the BaseBrain class (or one of its children).

    Raises
    ------
    None

    Returns
    -------
    cls : :py:class`~halsey.brains.basebrain.BaseBrain`
        The unaltered passed-in object.
    """
    brainRegister[cls.__name__] = cls
    return cls


# ============================================
#            register_conv_net
# ============================================
def register_conv_net(func):
    """
    This wrapper takes in a convolutional network construction function
    and adds it to the network register for ease of validation and
    network creation.

    .. note::

        The reason there are registers for each type of network is
        because RNNs require special treatment. They **MUST** use some
        kind of episodic memory and they MUST have the channels first in
        their input shapes. It's possible that the only separate
        register that's needed is one for RNNs, but since those aren't
        implemented yet, this will serve as a reminder.

    Parameters
    ----------
    func: function
        A convolutional neural network construction function.

    Raises
    ------
    None

    Returns
    -------
    func : function
        The unaltered passed-in network construction function.
    """
    networkRegister[func.__name__] = func
    convNetRegister[func.__name__] = func
    return func


# ============================================
#         register_experience_memory
# ============================================
def register_experience_memory(cls):
    """
    This wrapper takes in an experience memory object and adds it to
    the memory register for ease of validation and object creation.

    .. note::

        The reason there are registers for each type of network is
        because RNNs require special treatment. They **MUST** use some
        kind of episodic memory and they MUST have the channels first in
        their input shapes. It's possible that the only separate
        register that's needed is one for episodic memory, but since
        that isn't implemented yet, this will serve as a reminder.

    Parameters
    ----------
    cls : :py:class`~halsey.memory.basememory.BaseMemory`
        An instance of the BaseMemory class (or one of its children).

    Raises
    ------
    None

    Returns
    -------
    cls : :py:class`~halsey.memory.basememory.BaseMemory`
        The unaltered passed-in object.
    """
    memoryRegister[cls.__name__] = cls
    experienceMemoryRegister[cls.__name__] = cls
    return cls


# ============================================
#           register_action_manager
# ============================================
def register_action_manager(cls):
    """
    This wrapper takes in an action manager object and adds it to the
    action manager register for ease of validation and object creation.

    Parameters
    ----------
    cls : :py:class`~halsey.actions.basechooser.BaseChooser`
        An instance of the BaseChooser class (or one of its
        children).

    Raises
    ------
    None

    Returns
    -------
    cls : :py:class`~halsey.actions.basechooser.BaseChooser`
        The unaltered passed-in object.
    """
    actionManagerRegister[cls.__name__] = cls
    return cls


# ============================================
#            register_navigator
# ============================================
def register_navigator(cls):
    """
    This wrapper takes in a navigator object and adds it to the
    navigation register for ease of validation and object creation.

    Parameters
    ----------
    cls : :py:class`~halsey.navigation.basenavigator.BaseNavigator`
        An instance of the BaseNavigator class (or one of its
        children).

    Raises
    ------
    None

    Returns
    -------
    cls : :py:class`~halsey.navigation.basenavigator.BaseNavigator`
        The unaltered passed-in object.
    """
    navigatorRegister[cls.__name__] = cls
    return cls


# ============================================
#           register_frame_manager
# ============================================
def register_frame_manager(cls):
    """
    This wrapper takes in a frame manager object and adds it to the
    frame manager register for ease of validation and object creation.

    Parameters
    ----------
    cls : :py:class`~halsey.frames.baseprocessor.BaseFrameManager`
        An instance of the BaseFrameManager class (or one of its
        children).

    Raises
    ------
    None

    Returns
    -------
    cls : :py:class`~halsey.frames.baseprocessor.BaseFrameManager`
        The unaltered passed-in object.
    """
    frameManagerRegister[cls.__name__] = cls
    return cls


# ============================================
#              validate_params
# ============================================
def validate_params(params):
    """
    Makes sure that every passed parameter is both known to Halsey as
    well as the correct type.

    Parameters
    -----------
    params : dict
        The raw parameters as read in from the parameter file.

    Raises
    -------
    None

    Returns
    -------
    None
    """
    pass


# ============================================
#                is_empty_dir
# ============================================
def is_empty_dir(directory):
    """
    Checks to see if the output directory tree is empty.

    This is to prevent overwriting a directory already in use by
    another run.

    Parameters
    ----------
    directory : str
        The root of the output directory tree.

    Raises
    ------
    None

    Returns
    -------
    isEmpty : bool
        If True then the directory tree is empty and we can proceed.
    """
    isEmpty = True
    # Walk the directory tree
    for root, dirs, files in os.walk(directory):
        # If there are files, then we're done
        if len(files) != 0:
            isEmpty = False
            break
    return isEmpty
