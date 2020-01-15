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
#              get_mode_params
# ============================================
def get_mode_params(modeDict):
    """
    Extracts only those parameters related to a chosen option in a
    nested section of the parameter file.

    The parameter file is nested. That is, certain sections have
    user-selectable options (such as the learning method to use
    or the type of memory). The use of the mode keyword allows for
    a swtich-like interface to select these options. Those parameters
    listed under the option with `enabled=True` are the ones read in
    here.

    Parameters
    ----------
    modeDict : dict
        A dictionary of all the parameters listed under the `mode`
        key for a section in the parameter file.

    Raises
    ------
    None

    Returns
    -------
    modeParams : dict
        The parameters under the selected option (i.e., the parameters
        under the key with `enabled=True`.
    """
    # This holds the name of the chosen mode (e.g., the learning
    # technique to use) as well as any parameters specific to that
    # mode (e.g., fixedQSteps for the fixedQ learning technique)
    modeParams = {}
    # There needs to be exactly one mode set
    modeTot = 0
    for key in modeDict.keys():
        if modeDict[key]["enabled"]:
            modeParams = modeDict[key]
            modeParams.update({"mode": key})
            del modeParams["enabled"]
            modeTot += 1
    if modeTot != 1:
        raise
    return modeParams


# ============================================
#               get_new_folio
# ============================================
def get_new_folio(clArgs, params):
    """
    Converts the dictionary version of the parameters read in from the
    parameter file to a Folio.

    Parameters
    ----------
    clArgs : argparse.Namespace
        The object containing the parsed command-line arguments.

    params : dict
        Contains the parameters read in from the parameter file.

    Raises
    ------
    None

    Returns
    -------
    folio : anna.utils.folio.Folio
        A container providing an object interface to the parameter file
        parameters and command-line arguments.
    """
    # Create an empty folio
    folio = Folio()
    # Convert parameters from dictionary format to class format
    for section, sectionParams in params.items():
        setattr(folio, section, Folio())
        if "general" in sectionParams.keys():
            folio.__dict__[section].__dict__.update(sectionParams["general"])
        if "mode" in sectionParams.keys():
            modeParams = get_mode_params(sectionParams["mode"])
            folio.__dict__[section].__dict__.update(modeParams)
    # Add the command-line arguments to the folio
    setattr(folio, "clArgs", clArgs)
    return folio
