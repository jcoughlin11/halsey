"""
Title:   reader.py
Author:  Jared Coughlin
Date:    8/22/19
Purpose: Contains the Reader class, which holds all functions related
         to reading in files.
Notes:
"""
import yaml

import anna


#============================================
#                   Reader
#============================================
class Reader:
    """
    Container for all functions related to reading in files.

    Attributes:
    -----------
        pass

    Methods:
    --------
        pass
    """
    #-----
    # Constructor
    #-----
    def __init__(self):
        pass

    #-----
    # read_parameter_file
    #-----
    def read_parameter_file(self, paramFile):
        """
        Reads in the parameter file into a dictionary.
        
        architecture    : string, the network architecture to use
        batchSize       : int, size of batches used for training
        ckptFile        : string, name of file to use for saving/loading
        cropBot         : int, num rows to chop off bottom of frame
        cropLeft        : int, num cols to chop off left of frame
        cropRight       : int, num cols to chop off right of frame
        cropTop         : int, num rows to chop off top of frame
        discount        : float, reward discount rate
        enableDoubleDqn : bool, True if using double DQN
        enableFixedQ    : bool, True if using fixed-Q
        enablePer       : bool, True if using prioritized exp replay,
        env             : string, name of the gym environment to use
        epsDecayRate    : float, rate the  explore-exploit param decays
        epsilonStart    : float, start val of explore-exploit parameter
        epsilonStop     : float, min value of explore-exploit parameter
        fixedQSteps     : int, steps between weight copies w/ fixed-Q
        learningRate    : float, network learning rate
        loss            : string, the loss function to minimize
        maxEpisodeSteps : int, max number of steps per episode
        memorySize      : int, max number of experiences to store in
                              memory buffer
        nEpisodes       : int, number of episodes to train for
        nStackedFrames  : int, number of frames to stack
        optimizer       : string, name of the optimizer to use
        perA            : float, alpha parameter in eq. 1 of Schaul16
        perB            : float, beta param in IS weights of Schaul16
        perBAnneal      : float, annealment rate of IS weights
        perE            : float, epsilon parameter in prop. prior. of
                              Schaul16
        pretrainLen     : int, num experiences to initially fill mem
        pretrainNEps    : int, Num eps to use when prepopulating RNN
        renderFlag      : bool, Render scene during testing if True
        savePath        : string, path of checkpoint and param file
        savePeriod      : int, save model every savePeriod episodes
        shrinkCols      : int, y size of shrunk frame
        shrinkRows      : int, x size of shrunk frame
        testFlag        : bool, Test the agent if True
        traceLen        : int, num connected frames in RNN sample
        timeLimit       : int, max number of seconds to run for
        trainFlag       : bool, Train network if True

        Parameters:
        -----------
            paramFile : string
                The name of the parameter file to read (yaml).

        Raises:
        -------
            pass

        Returns:
        --------
            params : dict
                A dictionary keyed by the parameter names.
        """
        # Read data from parameter file
        with open(paramFile, 'r') as f:
            params = yaml.load(f)
        # Do a validation check on the parameters
        anna.nnio.ioutils.validate_params(params)
        # Check for option conflicts
        anna.nnio.ioutils.conflict_check(params)
        return params
