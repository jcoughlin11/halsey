"""
Title:   brick.py
Author:  Jared Coughlin
Date:    3/18/19
Purpose: Use DQN to teach an agent to play brick!
Notes:  1. This is based on the version in ../../course_version/, which, in turn, is from:
            https://tinyurl.com/ya8d9wcd
"""
import sys

import gym
import tensorflow as tf

import gym_brick
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
env = gym.make('brick-v0')

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
