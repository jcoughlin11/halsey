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
import tensorflow as tf

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
tf.reset_default_graph()
with tf.Session() as sess:
    dqn = nw.DQNetwork(hyperparams, env, 'dqn', sess)

    # Train the network
    if hyperparams['train_flag']:
        print('Training...')
        dqn.train(hyperparams['restart_training'])

    # Test the network
    print('Testing agent...')
    dqn.test(hyperparams['render_flag'])
