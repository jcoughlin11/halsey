"""
Title: misc.py
Notes:
"""
import os
import shutil

from .devices import using_gpu
from .endrun import endrun


# ============================================
#                 lock_file
# ============================================
def lock_file(fileName, path):
    """
    Doc string.
    """
    lockFile = os.path.join(path, ".".join([fileName.split(".")[0], "lock"]))
    shutil.copyfile(fileName, lockFile)
    os.chmod(lockFile, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


# ============================================
#           create_output_directory 
# ============================================
def create_output_directory(path):
    """
    Doc string.
    """
    try:
        os.path.makedirs(path)
    # Directory already exists
    except OSError:
        # If it's not empty we don't want to overwrite
        if len(os.listdir(path)) != 0:
            msg = f"Output directory `{path}` isn't empty."
            endrun(msg)


# ============================================
#                sanitize_path
# ============================================
def sanitize_path(path):
    """
    Doc string.
    """
    return os.abspath(os.expanduser(os.expandvars(path)))


# ============================================
#               get_data_format
# ============================================
def get_data_format():
    """
    Doc string.
    """
    channelsFirst = False
    if using_gpu():
        channelsFirst = True
    return channelsFirst
