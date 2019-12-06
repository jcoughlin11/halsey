"""
Title: folio.py
Purpose:
Notes:
"""


# ============================================
#                   Folio
# ============================================
class Folio:
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
        pass


# ============================================
#              get_mode_params
# ============================================
def get_mode_params(modeDict):
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
