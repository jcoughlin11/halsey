"""
Title: base.py
Notes:
"""
from abc import ABC
from abc import abstractmethod


# ============================================
#                 BaseNetwork
# ============================================
class BaseNetwork(ABC):
    """
    The `network`(s) contain the weights. These weight parameters are
    tuned during the learning process in order to increase the agent's
    performance. The weights are also arranged into a specific
    architecture, which is also held by this object. Both the value and
    architecture of the weights influence performance.
    """
