"""
Title: main.py
Notes:
"""
from halsey.utils.setup import setup


# ============================================
#                    run
# ============================================
def run():
    """
    Doc string.
    """
    manager = setup()
    if manager.doTraining:
        manager.train()
