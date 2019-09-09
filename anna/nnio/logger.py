"""
Title:   logger.py
Purpose: Contains the Logger class
Notes:
"""
import os


#============================================
#                  Logger
#============================================
class Logger:
    """
    Used for storing important information about a training and
    testing run.

    Attributes:
    -----------
        pass

    Raises:
    -------
        pass

    Returns:
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
        self.baseDir  = None
        self.fileBase = None

    #-----
    # set_params
    #-----
    def set_params(self, params):
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
        self.baseDir  = os.path.join(params['outputDir'], 'logs')
        self.fileBase = params['fileBase']
