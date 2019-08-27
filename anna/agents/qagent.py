"""
Title:   qagent.py
Author:  Jared Coughlin
Date:    8/22/19
Purpose: Contains the Agent class for learning via Deep-Q Learning
Notes:
"""


#============================================
#                   Agent
#============================================
class Agent:
    """
    Container and manager for learning to play games based on screen
    input via DQL.

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
    def __init__(self, args) 
        """
        Parameters:
        -----------
            args : argparse.Namespace
                Class whose attributes are the known command line
                arguments. See ioutils.parse_args() in anna.nnio.
        Raises:
        -------
            pass

        Returns:
        --------
            None
        """
        # Set up object for handling I/O
        self.ioManager = anna.nnio.manager.IOManager()
        # Read in parameter file
        self.params = self.ioManager.reader.read_parameter_file(args.paramFile)
        # Set up the memory
        self.memory = anna.memory.utils.init_memory(self.params)

    #-----
    # train
    #-----
    def train(self):
        """
        Manager for deciding whether to train or not.

        Parameters:
        -----------
            None

        Raises:
        -------
            pass

        Returns:
        --------
            None
        """
        if self.params['trainFlag']:
            pass

    #-----
    # test
    #-----
    def test(self):
        """
        Manager for deciding whether to test the agent or not.

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
        if self.params['testFlag']:
            pass
