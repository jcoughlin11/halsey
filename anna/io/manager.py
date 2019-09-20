"""
Title:   manager.py
Purpose: Contains the IoManager class, which is a convenience object
            for holding the Reader, Writer, and Logger objects.
Notes:
"""
from . import reader
from . import writer


#============================================
#                 IoManager
#============================================
class IoManager:
    """
    A convenience object for holding the Writer and Reader objects.

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
        """
        Doc string.

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
        self.reader = reader.Reader()
        self.writer = writer.Writer()

    #-----
    # set_params
    #-----
    def set_params(self, ioParams):
        """
        Doc string.

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
        self.reader.set_params(ioParams)
        self.writer.set_params(ioParams)
