"""
Title:   manager.py
Author:  Jared Coughlin
Date:    8/22/19
Purpose: Contains the IOManager class for handling all file reading and
         writing.
Notes:
"""


#============================================
#                  IOManager
#============================================
class IOManager:
    """
    Primary manager for reading and writing data to and from files.

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
        """
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
        # Set up objects for reading and writing files
        self.reader = anna.nnio.reader.Reader()
        self.writer = anna.nnio.writer.Writer()
        self.logger = anna.nnio.logger.Logger()
