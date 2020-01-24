"""
Title: folio.py
Purpose: Contains the Folio object as well as functions related to
            managing it.
Notes:
"""


# ============================================
#                   Folio
# ============================================
class Folio:
    """
    An empty container object that provides an object interface to data
    normally saved in a dictionary.

    This is a convenience object, since the object interface is cleaner
    than the dictionary interface.

    Attributes
    ----------
    None

    Methods
    -------
    None
    """

    # -----
    # constructor
    # -----
    def __init__(self):
        pass


# ============================================
#               get_new_folio
# ============================================
def get_new_folio(params):
    """
    Converts the dictionary version of the parameters read in from the
    parameter file to a Folio.

    The parameter file has the form:

    sectionName1:
        param1 : value
        param2 : value
        option1:
            enabled : True/False
            option1_param1 : value
            ...
        option2:
            enabled : True/False
            ...
    sectionName2:
        ...

    We only want to keep the parameters from the options that are
    enabled.

    Parameters
    ----------
    params : dict
        Contains the parameters read in from the parameter file.

    Raises
    ------
    None

    Returns
    -------
    folio : halsey.utils.folio.Folio
        A container providing an object interface to the parameter file
        parameters and command-line arguments.
    """
    # Create an empty folio
    folio = Folio()
    # Convert parameters from dictionary format to class format
    for section, sectionParams in params.items():
        setattr(folio, section, Folio())
        # Loop over each of the section's parameters
        for paramName, paramVal in sectionParams.items():
            # Handle an option
            if isinstance(paramVal, dict):
                # Only keep the option around if it's enabled
                if paramVal["enabled"]:
                    folio.__dict__[section].__dict__.update(paramVal)
                    del folio.__dict__[section].__dict__["enabled"]
                    folio.__dict__[section].__dict__.update({"mode": paramName})
            else:
                folio.__dict__[section].__dict__.update({paramName: paramVal})
    return folio


# ============================================
#              finalize_folio
# ============================================
def finalize_folio(inputShape, nActions, folio):
    """
    This function adds any parameters that are needed in multiple
    sections of the folio to those sections.

    While this results in potentially multiple copies of several
    variables, having each section of the folio be isolated from every
    other section is enormously convenient. All of the variables here
    are small, so the impact on memory usage and performance should be
    negligible.

    Parameters
    ----------
    inputShape : list
        The dimensions of the input to the neural network(s).

    nActions : int
        The size of the action space for the game being learned.

    folio : halsey.utils.folio.Folio
        The folio to be finalized.

    Raises
    ------
    None

    Returns
    -------
    finalFolio : halsey.utils.folio.Folio
        The version of the folio containing the fully self-contained
        sections.
    """
