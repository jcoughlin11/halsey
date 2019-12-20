"""
Title: validation.py
Purpose:
Notes:
"""


# ============================================
#                 Registers
# ============================================
# Run parameters
runRegister = {"envName": str, "train": bool, "test": bool, "timeLimit": int}

# Io parameters
ioRegister = {"outputDir": str, "fileBase": str}

# Training parameters
trainingRegister = {
    "nEpisodes": int,
    "maxEpisodeSteps": int,
    "batchSize": int,
    "savePeriod": int,
}

trainersRegister = [
    "qtrainer",
]

trainingRegister["trainers"] = trainersRegister

# Brain parameters
brainRegister = {"discount": float, "learningRate": float}

convRegister = [
    "conv1",
]

duelingRegister = [
    "dueling1",
]

rnnRegister = [
    "rnn1",
]

lossRegister = []

# Memory parameters
memoryRegister = {
    "maxSize": int,
    "pretrainLen": int,
}

# Action parameters
actionRegister = []

# Navigation parameters
navigationRegister = []

# Frame parameters
frameRegister = []

registers = {
    "run": runRegister,
    "io": ioRegister,
    "training": trainingRegister,
    "brain": brainRegister,
    "memory": memoryRegister,
    "action": actionRegister,
    "navigation": navigationRegister,
    "frame": frameRegister,
}


# ============================================
#             validate_params
# ============================================
def validate_params(folio):
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
    for section, sectionParams in registers.items():
        # Make sure every required section is present in the folio
        if section not in folio.__dict__.keys():
            raise
        # Now check each section's parameters
        for paramName, paramValue in sectionParams.items():
            pass
