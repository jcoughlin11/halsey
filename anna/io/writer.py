"""
Title:   writer.py
Purpose: Contains the writer class
Notes:
"""


#============================================
#                  Writer
#============================================
class Writer:
    """
    Used to write data to files.

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
        self.baseDir  = params['outputDir']
        self.fileBase = params['fileBase']
