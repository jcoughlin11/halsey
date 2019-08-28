"""
Title:   loader.py
Author:  Jared Coughlin
Date:    8/28/19
Purpose: Contains the Loader class used for reading in data
Notes:
"""


#============================================
#                   Loader
#============================================
class Loader:
    """
    Container for functions used to load in already saved data.

    Attributes:
    -----------
        pass

    Methods:
    --------
        pass
    """
    #-----
    # constructor
    #-----
    def __init__(self):
        pass

    #-----
    # load_train_params
    #-----
    def load_train_params(params):
        """
        Loads in the core parameters used in the training loop in order
        to continue training from where it left off.

        startEp : The value of the current episode
        decayStep : The value of the decay step used in epsilon-greedy
        step : The value of the current intra-episode step
        fixedQStep : The number of steps since the tnet was updated
        totalRewards : List of each completed episode's total reward
        epRewards : List of rewards doled out for the current episode
        state : The current state (np array)
        frameStack : deque version of the state

        Parameters:
        -----------
            pass

        Raises:
        -------
            pass

        Returns:
        --------
            pass
        """
        # Set up hdf5 file
        fname = os.path.join(savePath, 'training_params.h5')
        # Read hdf5 file
        with h5py.File(fname, "r") as h5f:
            # Load counters
            counters = list(h5f["counters"][:])
            # Rewards
            totRewards = list(h5f["totrewards"][:])
            epRewards = list(h5f["eprewards"][:])
            # State (it's actually the stacked set of processed frames)
            state = h5f["state"][:]
            # Unpack the stacked state such that each individual frame
            # in the stack is an element of the deque
            frameStack = UNPACK_STATE
        # Package the training parameters into a class
        trainParams = anna.training.utils.TrainParams(counters, totRewards, epRewards, state, frameStack)
        return trainParams
