"""
Title: registry.py
Notes:
    * https://tinyurl.com/yydsl4jq
"""
registry = {}


# ============================================
#                   register
# ============================================
def register(cls):
    """
    Adds the given class to the `registry` dictionary.

    Called by the `__init_subclass__` hook in each parent class. This
    allows for classes to be self-registering when they are defined.

    This method makes it so that each individual class does not have to
    be a gin configurable; only the setup functions do, which makes
    the maintenence process easier.
    """
    registry[cls.__name__] = cls
