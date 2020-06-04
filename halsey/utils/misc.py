"""
Title: misc.py
Notes:
"""
import os
import shutil
import stat

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
        os.makedirs(path)
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
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))


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


# ============================================
#                   io_check
# ============================================
def io_check(outputDir):
    """
    Makes sure output dir can be made if starting from scratch
    or that output dir exists and has all the right files in it
    if continuing training or testing or analyzing. That kind of
    stuff. This prevents lots of time being spent on something only
    to find out that none of the work can be saved for some reason.
    Better to find that out before doing all the work.
    """
    create_output_directory(outputDir)
