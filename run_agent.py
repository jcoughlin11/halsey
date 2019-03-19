"""
Title:   run_agent.py
Author:  Jared Coughlin
Date:    3/19/19
Purpose: Driver code for using DQL to train an agent to play a game
Notes:  1. This is based on the version in ../../course_version/, which, in turn, is from:
            https://tinyurl.com/ya8d9wcd
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
env = gym.make(hyperparams['env_name'])

# Set up the network
print('Setting up network...')
tf.reset_default_graph()
with tf.Session() as sess:
    agent = nw.Agent(hyperparams, env, 'agent', sess)

    # Train the network
    if hyperparams['train_flag']:
        print('Training...')
        agent.train(hyperparams['restart_training'])

    # Test the network
    if hyperparams['test_flag']:
        print('Testing agent...')
        agent.test(hyperparams['render_flag'])
