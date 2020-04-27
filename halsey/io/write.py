"""
Title: write.py

Notes:
"""
import os
import shutil
import stat

import yaml


# ============================================
#              save_checkpoint
# ============================================
def save_checkpoint(instructor):
    """
    Doc string.
    """
    # Save network(s) and optimizer(s)
    instructor.checkpointManager.save()
    # Save instructor's state (e.g., current episode, etc.)
    save_instructor_state(instructor)
    # Save the environment-specific variables
    save_navigator(instructor.navigator)


# ============================================
#           lock_parameter_file
# ============================================
def lock_parameter_file(paramFile):
    """
    Doc string.
    """
    paramLockFile = os.path.join(os.getcwd(), "params.lock")
    shutil.copyfile(paramFile, paramLockFile)
    os.chmod(paramLockFile, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


# ============================================
#          save_instructor_state
# ============================================
def save_instructor_state(instructor):
    """
    Doc string.
    """
    instructorState = instructor.get_instructor_state()
    fname = os.path.join(os.getcwd(), "instructor_state.yaml")
    with open(fname, "w") as f:
        yaml.safe_dump(instructorState, f)


# ============================================
#              save_navigator
# ============================================
def save_navigator(navigator):
    """
    Doc string.
    """
    raise NotImplementedError
