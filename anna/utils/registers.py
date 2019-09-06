"""
Title:   registers.py
Purpose: Contains registers for all recognized parameters and command
            line arguments.
Notes:
"""


# Register for sections in the parameter file
paramFileSectionRegister = {'run', 'io', 'training', 'network', 'memory', 'explore', 'frame'} 

# Run register
paramRunRegister = {'env', 'render', 'test', 'timeLimit', 'train'}

# IO register
paramIoRegister = {'ckptBase', 'outputDir', 'savePeriod'}

# Training register
paramTrainingRegister = {'batchSize', 'discount', 'learningRate', 'maxEpisodeSteps'}

# Network register
paramNetworkRegister = {'architecture', 'loss', 'optimizer', 'mode'}

# Memory register
paramMemoryRegister = {'memorySize', 'pretrainLen', 'priority', 'mode'}

# Explore register
paramExploreRegister = {'epsilonGreedy'}

# Frame register
paramFrameRegister = {'cropBot', 'cropLeft', 'cropRight', 'cropTop', 'shrinkCols', 'shrinkRows', 'traceLen'}
