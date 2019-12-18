"""
Title: endrun.py
Purpose:
Notes:
"""
import os


# ============================================
#             check_early_stop
# ============================================
def check_early_stop():
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
    earlyStop = False
    if os.path.isfile(os.path.join(os.getcwd(), "stop")):
        earlyStop = True
    return earlyStop
