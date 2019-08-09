"""
Title:   nnutils.py
Author:  Jared Coughlin
Date:    1/24/19
Purpose: Contains utility and helper classes related to neural networks
Notes:
"""
import os
import subprocess32
import sys
import time

import gym
import numpy as np
import tensorflow as tf

import losses
import nnio as io


#============================================
#                initialize
#============================================
def initialize():
    """
    Reads the parameter file and sets up the gym environment.

    Parameters:
    -----------
        None

    Raises:
    -------
        None

    Returns:
    --------
        hyperparams : dict
            Dictionary containing the parameters read in from the file.

        env : gym.core.Env
            The object that interfaces between the game and the agent.
    """
    # Read hyperparameters from parameter file
    try:
        print("Reading hyperparameters...")
        hyperparams = io.read_hyperparams(sys.argv[1])
    except (IOError, IndexError):
        print("Error, could not open file for reading hyperparameters!")
        sys.exit(1)
    # Create the gym environment
    try:
        print("Building the environment...")
        env = gym.make(hyperparams["env_name"])
    except KeyError:
        print("Error, could not find 'env_name' in the hyperparams!")
        sys.exit(1)
    except gym.error.UnregisteredEnv:
        print("Error, trying to build an unregistered environment:"
            "\n\t{}\n"
            "Try:\n\tfrom gym.envs.registration import register\n"
            "\tregister(\n"
            "\t\tid='<env-name>',\n"
            "\t\tentry_point='<package>.envs:<EnvClass>',\n"
            "\t)\n"
            "In the environment's init.py".format(hyperparams['env_name'])
        )
        sys.exit(1)
    except gym.error.DeprecatedEnv as e:
        print("Error, trying to build a deprecated environment:\n"
            "\t{}\n"
            "{}".format(hyperparams['env_name'], e)
        )
        sys.exit(1)
    except gym.error.Error:
        print("Error, trying to build an unrecognized environment!")
        sys.exit(1)
    return hyperparams, env


#============================================
#             check_early_stop
#============================================
def check_early_stop(savePath):
    """
    Looks for a file called stop in the current working directory.

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
    stopFile = os.path.join(savePath, 'stop')
    if os.path.isfile(stopFile):
        # Remove the file
        subprocess32.call(["rm", stopFile])
        return True
    else:
        return False


#============================================
#            time_limit_reached
#============================================
def time_limit_reached(start, limit):
    """
    Checks to see whether or not the max allowed training time has
    elapsed. 
    """
    if time.time() - start > limit:
        return True
    else:
        return False


#============================================
#                  set_loss
#============================================
def set_loss(params):
    """
    If applicable, assigns the string version of the loss function to
    the actual function version. Some strings, such as 'mse', are ok
    since tf recognizes them. For custom loss functions, such as with
    PER, it must be set explicitly.

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
    # Check for custom loss functions. If we're here, then the loss has
    # already been confirmed to be in the lossRegister and, therefore,
    # valid and listed here
    if params['loss'] == 'per_mse':
        params['loss'] = losses.per_mse
    return params


#============================================
#               set_optimizer
#============================================
def set_optimizer(params):
    """
    Converts the string form of the optimizer to the class form and
    applies the learning rate.

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
    if params['optimizer'] == 'adam':
        params['optimizer'] = tf.keras.optimizers.Adam(
            learning_rate=params['learning_rate']
        )
    return params


#============================================
#               Sumtree Class
#============================================
class SumTree:
    """
    Prioritized experience replay makes use of a sum tree to efficiently
    store and fetch data. A sum tree is a binary tree where the value of
    each node is the sum of the values in its child nodes. Here, the
    actual priorities are stored in the leaf nodes of the tree. This is
    an unsorted tree.
    
    Assuming a perfectly balanced tree, the number of nodes in the tree
    is  nNodes = 2 * nLeafs - 1. This is because, in a binary tree, the
    number of nodes at a given level is twice the number of nodes in the
    level before it. We then have to subract off 1 because at the root
    level there is only one node. This assumes a  perfectly balanced
    tree (that is, every node has both a left and right child and that
    each subtree descends to the same level), i.e.,

                                    o
                                   / \
                                  o   o
                                 / \ / \
                                o  o o  o

    The nodes are stored in an array level-wise. That is, root is index
    0, root's left child is at index 1, root's right child is at index
    2, then we go to the next level down and go across left to right. As
    such, the indices of the leaf nodes start at nLeafs - 1 and continue
    on to the end of the array.

    Attributes:
    -----------
        dataPointer : int
            The leaves of the tree are filled from left to right. This
            is an index that keeps track of where we are in the leaf
            row of the tree.

        tree : ndarray
            This is an array used to store the actual sum tree.

        data : ndarray
            Attached to each leaf is a priority (stored in the tree
            attribute) as well as the actual experience that has that
            priority. This array holds the experience tuples.

    Methods:
    --------
        pass
    """

    # -----
    # Constructor
    # -----
    def __init__(self, nLeafs):
        """
        Parameters:
        -----------
            nLeafs : int
                The number of leaf nodes the tree will have. This is
                equal to the number of experiences we want to store,
                since the priorities for each experience go into the
                leaf nodes.

        Raises:
        -------
            pass

        Returns:
        --------
            pass
        """
        self.nLeafs = nLeafs
        self.dataPointer = 0
        self.tree = np.zeros(2 * self.nLeafs - 1)
        self.data = np.zeros(self.nLeafs, dtype=object)

    # -----
    # add
    # -----
    def add(self, data, priority):
        """
        Takes in an experience as well as the priority assigned to that
        experience, and assigns it to a leaf node in the tree,
        propagating the the changes throughout the rest of the tree.

        Parameters:
        -----------
            priority : float
                The priority that has been assigned to this particular
                experience.

            data : tuple
                The experience tuple being added to the tree.

        Raises:
        -------
            pass

        Returns:
        --------
            None
        """
        # Get the index of the array corresponding to the current leaf
        # node
        tree_index = self.dataPointer + self.nLeafs - 1
        # Insert the experience at this location (like the deque, this
        # starts to 'forget' experiences once the max capacity is
        # reached, i.e., they are overwritten)
        self.data[self.dataPointer] = data
        # Update the tree
        self.update(tree_index, priority)
        # Advance to the next leaf
        self.dataPointer += 1
        # If we're above the max value, then we go back to the beginning
        if self.dataPointer >= self.nLeafs:
            self.dataPointer = 0

    # -----
    # update
    # -----
    def update(self, tree_index, priority):
        """
        Handles updating the current leaf's priority score and then
        propagating that change throughout the rest of the tree.

                                            0
                                           / \
                                          1   2
                                        /  \  / \
                                       3   4  5  6

        The numbers above are the indices of the nodes. If the tree
        looks like:

                                            48
                                           / \
                                          13   35
                                        /  \  / \
                                       4   9  33  2

        and the value of node 6 changes to 8, the new sumtree will look
        like:

                                            54
                                           / \
                                          13   41
                                        /  \  / \
                                       4   9  33  8

        That is, we need to update the value of each of the changed
        node's ancestors. We get at the parent node by doing
        (currentNodeIndex - 1) // 2. E.g., (6-1) // 2 = 2, then
        (2-1)//2 = 0, which gives us the indices of the two nodes we
        would need to update in this case (2 and 0). The update is to
        simply add the change made to the current node to the value of
        it's parent. E.g., here node 6 has change = 8 - 2 = 6, so the
        update to node 2 is: change2 = 35 + change = 41. We then repeat
        this process until we've updated root.

        Parameters:
        -----------
            tree_index : int
                The index of self.tree that corresponds to the current
                leaf node.

            priority : float
                The value of the priority to assign to the current leaf
                node.

        Raises:
        -------
            pass

        Returns:
        --------
            None
        """
        # Get the difference between the new priority and the old
        # priority
        deltaPriority = priority - self.tree[tree_index]
        # Update the node with the new priority
        self.tree[tree_index] = priority
        # Propogate the change throughout the rest of the tree (we're
        # done after we update root, which has an index of 0)
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += deltaPriority

    # -----
    # get_leaf
    # -----
    def get_leaf(self, value):
        """
        This function returns the experience whose priority is the
        closest to the passed value.

                                            48
                                           / \
                                          13   35
                                        /  \  / \
                                       4   9  33  2

        For the above tree (the numbers are the priorities, not
        indices), let's say we're in the situation where batchSize = 6,
        so we've broken up the range [0,48] into 6 equal ranges. If
        we're on the first one, it will span [0,8). We choose a random
        number from there, which is value, so let's say we get
        value = 7. This function would then return node 4, which has a
        value of 9 and is the closest to the passed value of 7.

        Parameters:
        -----------
            value : float
                We want to find the experience whose priority is the
                closest to this number.

        Raises:
        -------
            pass

        Returns:
        --------
            index : int
                The index of the tree corresponding to the chosen
                experience.

            priority : float
                The priority of the chosen experience.

            experience : tuple
                The experience whose priority is closest to value.
        """
        # Start at root
        parentIndex = 0
        # Search the whole tree
        while True:
            # Get the indices of the current node's left and right
            # children
            leftIndex = 2 * parentIndex + 1
            rightIndex = leftIndex + 1
            # Check exit condition
            if leftIndex >= len(self.tree):
                leafIndex = parentIndex
                break
            # Otherwise, continue the search
            else:
                if value <= self.tree[leftIndex]:
                    parentIndex = leftIndex
                else:
                    value -= self.tree[leftIndex]
                    parentIndex = rightIndex
        # Get the experience corresponding to the selected leaf
        dataIndex = leafIndex - self.nLeafs + 1
        return leafIndex, self.tree[leafIndex], self.data[dataIndex]

    # -----
    # total_priority
    # -----
    @property
    def total_priority(self):
        """
        This function returns the root node of the tree, which is just
        the sum of all of the priorities. The property decorator lets
        us access it as if it were an attribute.

        Parameters:
        -----------
            None

        Raises:
        -------
            pass

        Returns:
        --------
            totalPriority : float
                The sum of each leaf's priority in the tree, which is
                just the root node since this is a sum tree.
        """
        return self.tree[0]
