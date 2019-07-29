"""
Title:   run_agent.py
Author:  Jared Coughlin
Date:    3/19/19
Purpose: Driver code for using DQL to train an agent to play a game
Notes:  1. This is based on https://tinyurl.com/ya8d9wcd
"""
import sys

import gym
import tensorflow as tf

import agent
import error
import nnutils as nu


#============================================
#                   main
#============================================
def main():
    """
    Driver for training, testing, or running an agent instance.

    Parameters:
    -----------
        None

    Raises:
    -------
        None

    Returns:
    --------
        None
    """
    # Initialize the run
    hyperparams, env = nu.initialize()
    # Set up the network
    print("Setting up network...")
    tf.reset_default_graph()
    with tf.Session() as sess:
        ag = agent.Agent(hyperparams, env, sess)
        # Train the network
        if hyperparams["train_flag"]:
            print("Training...")
            ag.train(hyperparams["restart_training"])
        # Test the network
        if hyperparams["test_flag"]:
            print("Testing agent...")
            ag.test(hyperparams["render_flag"])
