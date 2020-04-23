"""
Title: write.py

Notes:
"""


# ============================================
#              save_checkpoint
# ============================================
def save_checkpoint(instructor):
    """
    Doc string.
    """
    # Save network(s) and optimizer(s)
    instructor.checkpointManager.save()
