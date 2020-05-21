"""
Title: base.py
Notes:
"""


# ============================================
#                 BaseBrain
# ============================================
class BaseBrain:
    """
    Doc string.
    """

    # -----
    # constructor
    # -----
    def __init__(self, nets, params):
        """
        Doc string.
        """
        self.optimizerName = params["optimizer"]
        self.lossName = params["loss"]
        self.nets = nets
