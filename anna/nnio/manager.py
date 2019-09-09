"""
Title:   manager.py
Purpose: Contains the IoManager class, which is a convenience object
            for holding the Reader, Writer, and Logger objects.
Notes:
"""
from . import logger
from . import reader
from . import writer


#============================================
#                 IoManager
#============================================
class IoManager:
    """
    A convenience object for holding Writer, Reader, and Logger
    objects.

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
        self.logger = logger.Logger()
        self.reader = reader.Reader()
        self.writer = writer.Writer()

    #-----
    # set_params
    #-----
    def set_params(self, clArgs, params):
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
        self.logger.set_params(params)
        self.reader.set_params(params)
        self.writer.set_params(params)
