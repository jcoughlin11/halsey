"""
Title:   space_invaders.py
Author:  Jared Coughlin
Date:    1/23/19
Purpose: Use DQN to teach an agent to play space invaders!
Notes:  1. This is based on the version in ../../course_version/, which, in turn, is from:
            https://tinyurl.com/ya8d9wcd
        2. The other version used straight tensorflow, but was disorganized. This is an
            attempt to clean it up. Keras works great, but is too slow on a laptop for
            something like this.
"""
import sys

import gym
from tensorflow.keras.models import load_model

import nnetworks as nw
import nnutils as nu



# Read hyperparameters from parameter file
try:
    print('Reading hyperparameters...')
    hyperparams = nu.read_hyperparams(sys.argv[1])
except (IOError, IndexError) as e:
    print('Error, could not open file for reading hyperparameters!')
    sys.exit()

# Create the gym environment
print('Building the environment...')
env = gym.make('SpaceInvaders-v0')

# Set up the network
print('Setting up network...')
dqn = nw.DQNetwork(hyperparams, env)

# Train the network
if hyperparams['train_flag']:
    print('Training...')
    dqn.train()
    dqn.save('./models/space_invaders.ckpt')
else:
    print('Loading model...')
    # Try loading the desired model
    try:
        dqn.load('./models/space_invaders.ckpt')
    except ValueError:
        print('Error, could not open model file to load!')
        sys.exit()

# Test the network
print('Testing agent...')
dqn.test(hyperparams['render_flag'])
