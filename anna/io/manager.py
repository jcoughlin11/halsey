"""
Title:   manager.py
Purpose: 
Notes:
"""
import os

import anna


# ============================================
#                 IoManager
# ============================================
class IoManager:
    """
    Doc string.

    Attributes:
    -----------
        pass

    Methods:
    --------
        pass
    """

    # -----
    # constructor
    # -----
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
        self.reader = anna.io.reader.Reader()
        self.writer = anna.io.writer.Writer()

    # -----
    # load_params 
    # -----
    def load_params(self):
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
        # Parse the command-line args
        clArgs = self.reader.parse_args()
        # Read the parameter file
        params = self.reader.read_param_file(clArgs.paramFile, clArgs.continueTraining)
        # Build the folio
        folio = anna.common.folio.get_new_folio(clArgs, params)
        # Validate the parameters
        anna.common.validation.validate_params(folio)
        # Set the io parameters
        self.set_io_params()
        return folio

    #-----
    # save_params
    #-----
    def save_params(self):
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
    # load_checkpoint 
    #-----
    def load_checkpoint(self):
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
    # save_checkpoint
    #-----
    def save_checkpoint(self):
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

    # -----
    # set_io_params
    # -----
    def set_io_params(self):
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
