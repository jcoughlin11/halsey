"""
Title:   reader.py
Author:  Jared Coughlin
Date:    8/22/19
Purpose: Contains the Reader class, which holds all functions related
         to reading in files.
Notes:
"""
import yaml


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
        with open(paramFile, 'r') as f:
            params = yaml.load(f)
        return params
