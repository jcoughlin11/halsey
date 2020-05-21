"""
Title: main.py
Notes:
"""
from halsey.manager.manager import Manager
from halsey.utils.setup import setup


# ============================================
#                    run
# ============================================
def run():
    """
    Doc string.
    """
    clArgs = setup()
    manager = Manager(clArgs)
    manager.io_check()
    if manager.doTraining:
        manager.train()
