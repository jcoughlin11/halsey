"""
Title:   ioutils.py
Author:  Jared Coughlin
Date:    8/27/19
Purpose: Contains miscellaneous I/O functions
Notes:
"""
import argparse
import os


# Registers
archRegister = ["conv1", "dueling1", "rnn1"]
rnnRegister = ["rnn1"]
lossRegister = ["mse", "per_mse"]
optimizerRegister = ["adam"]
floatParams = [
    "discount",
    "epsDecayRate",
    "epsilonStart",
    "epsilonStop",
    "learningRate",
    "perA",
    "perB",
    "perBAnneal",
    "perE",
]
intParams = [
    "batchSize",
    "cropBot",
    "cropLeft",
    "cropRight",
    "cropTop",
    "fixedQSteps",
    "maxEpisodeSteps",
    "memorySize",
    "nEpisodes",
    "nStackedFrames",
    "pretrainLen",
    "pretrainNEps",
    "shrinkCols",
    "shrinkRows",
    "savePeriod",
    "traceLen",
    "timeLimit",
]
boolParams = [
    "enableDoubleDqn",
    "enableFixedQ",
    "enablePer",
    "renderFlag",
    "testFlag",
    "trainFlag",
]
stringParams = [
    "architecture",
    "ckpt_file",
    "env",
    "loss",
    "optimizer",
    "savePath",
]
typeRegisters = [
    [str, stringParams],
    [int, intParams],
    [float, floatParams]
]


#============================================
#                 parse_args
#============================================
def parse_args():
    """
    Sets up and collects the given command line arguments. These are:

        paramFile : string : required
            The name of the parameter file to read.

        --restart, -r : string : optional
            Flag indicating to restart training from the beginning
            using the given parameter file. If this is not present then
            the behavior defaults to looking for a restart file with
            which to continue training. If no restart file is found,
            the code begins a new training session with the given
            parameter file.

    Parameters:
    -----------
        None

    Raises:
    -------
        None

    Returns:
    --------
        args : argparse.Namespace
            A class whose attributes are the names of the known args
            given by calls to add_argument.
    """
    # Set up the parser
    parser = argparse.ArgumentParser()
    # Parameter file
    parser.add_argument(
        "paramFile",
        help="The name of the yaml file containing parameters for the run.",
    )
    # Restart flag
    parser.add_argument(
        "--restart",
        "-r",
        action="store_true",
        help="Restarts training using the given parameter file.",
    )
    args = parser.parse_args()
    return args


#============================================
#              validate_params
#============================================
def validate_params(params):
    """
    Makes sure that every parameter is recognized and has a valid
    value. 

    Parameters:
    -----------
        params : dict
            A dictionary containing the names and values of each
            parameter from the parameter file.

    Raises:
    -------
        pass

    Returns:
    --------
        pass
    """
    # Make sure each parameter is valid
    for k, v in params.items():
        validKey = False
        for r in typeRegisters:
            if k in r[1]:
                validKey = True
                if not isinstance(v, r[0]):
                    raise ValueError("Incorrect type for: {}".format(k))
                break
        if not validKey:
            raise ValueError("Unrecognized param: {}".format(k))


#============================================
#               conflict_check
#============================================
def conflict_check(params):
    """
    Checks for conflicts in the given parameters.

    Parameters:
    -----------
        params : dict
            Dictionary of params read in from the parameter file.

    Raises:
    -------
        pass

    Returns:
    --------
        pass
    """
    # Check to make sure the architecture has been defined
    if params["architecture"] not in archRegister:
        raise ValueError("Error, unrecognized network architecture!")
    # Check for valid loss function
    if params["loss"] not in lossRegister:
        raise ValueError("Error, unrecognized loss function!")
    # Check for valid optimizer function
    if params["optimizer"] not in optimizerRegister:
        raise ValueError("Error, unrecognized optimizer function!")
    # Double DQN requires fixed-Q
    if params["enableDoubleDqn"] and not params["enableFixedQ"]:
        raise ValueError("Error, double dqn requires the use of fixed Q!")
    # Make sure the save path exists. If it doesn't, try and make it
    if not os.path.exists(params["savePath"]):
        os.path.makedirs(params["savePath"])
    # If it does exist, make sure it's a directory
    elif not os.path.isdir(params["savePath"]):
        raise ValueError("savePath exists but is not a dir!")
    # Make sure either the train flag or test flag (or both) are set
    if not params["trainFlag"] and not params["testFlag"]:
        raise ValueError("Error, neither training nor testing enabled!")
    # Make sure we're using single frame stacks if using an RNN
    if params['architecture'] in rnnRegister and params['stackSize'] != 1:
        raise ValueError("Must use stack_size = 1 with an RNN!")
