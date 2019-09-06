"""
Title:   reader.py
Purpose: Contains the Reader class.
Notes:
"""
import yaml


#============================================
#                  Reader
#============================================
class Reader:
    """
    Used to read data from files.

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
        pass

    #-----
    # read_param_file
    #-----
    def read_param_file(self, paramFile):
        """
        Reads in the parameters from the given parameter file. See the
        README for a list and description of each parameter.

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
        with open(paramFile, 'r') as f:
            params = yaml.load(f)
        return params
