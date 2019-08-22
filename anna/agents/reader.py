"""
Title:   reader.py
Author:  Jared Coughlin
Date:    8/22/19
Purpose: Contains the Reader class, which holds all functions related
         to reading in files.
Notes:
"""


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
                The name of the parameter file to read.

        Raises:
        -------
            pass

        Returns:
        --------
            params : dict
                A dictionary keyed by the parameter names.
        """
        # Empty dictionary for holding params
        params = {}
        # Open the file for reading
        with open(paramFile, 'r') as f:
            # Loop over every line in the file
            for line in f:
                # Skip comments and empty lines
                if line[0] == '#' or not line.strip():
                    continue
                # Lines are: parameterName : parameterValue\n
                key, value = line.split(':')
                # Remove spaces from key
                key.strip()
                # Convert value to the proper type
                if key in anna.utils.ioutils.floatRegister:
                    value = float(value)
                elif key in anna.utils.ioutils.intRegister:
                    value = int(value)
                elif key in anna.utils.ioutils.stringRegister:
                    value = value.strip()
                else:
                    print("Parameter {} not found!".format(key))
                    sys.exit(1)
                params[key] = value
        return params
