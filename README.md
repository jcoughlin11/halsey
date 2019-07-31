# Introduction

This is ANNA (Arcade Neural Network Agent). It's a deep convolutional neural network built
on tensorflow and gym that's designed to learn how to play 2d games. The agent itself is
based off of the one given at [this site](https://simoninithomas.github.io/Deep_reinforcement_learning_Course/).

The agent is completely agnostic to the game it is learning. The only requirements
are that there be a gym environment for the game. Pretty much everything is controlled via the
parameter file.

# Reinforcement Learning

In **reinforcement learning (RL)**, there is an **agent** that interacts with an **environment**.
The agent performs **actions** in order to navigate through the environment. The environment
provides the agent with **rewards** R (positive and/or negative) based on the results of the
agent's actions. The agent's goal is to maximize its **return** G from the environment.

The return is some combination of the rewards provided to the agent by the environment. It allows
for specifying what the rewards mean to the agent (how they are used).

## Q Learning

ANNA is an agent that plays arcade games. Its return is the highest possible score
in the game. The rewards provided to ANNA from the environment are changes in game-score.
This means that, for each game state ANNA finds itself in, it needs to choose the action
that leads to the highest possible score.

One way to make this choice of action is to use a **Q-table**. This is a table that
contains scores, called Q-values, for each action in each state:

       | action1    action2    action3    ...
---------------------------------------------
state1 | Q_{s1,a1}  Q_{s1,a2}  Q_{s1,a3}  ...
state2 | ...
...    | ...

In order to get the highest score, the agent chooses the action with the highest Q-value
for the state that it currently finds itself in. The question then becomes:
how are the Q-values determined?

# Future Discounted Rewards

The sequence of states visited by the agent due to its action choices is called a
**trajectory**. Each action choice in a given state yields a reward from the environment.
Since ANNA's goal is to get the highest possible score in the game, and the rewards
given to ANNA by the environemnt are changes in the game score, this means that ANNA
wants to choose actions that, over the course of the trajectory, give the largest total
reward.

Therefore, one possible way to determine the return is to add up all of the rewards:

G = \sum_i R_i

where the sum is over each of the n states in the trajectory.

This has a problem, however: it treats all rewards equally, temporally speaking. That is,
the first reward given has the same importance as the last reward given. As an
example as to why this isn't ideal, consider someone with only 100 dollars whose goal is
to make the most amount of money possible. One way to do this is to invest that money
into a fund that gives a certain amount of compounding interest. While this will certainly
yield a large amount of money in the long-term, it doesn't address the issue of having
money for immediate concerns, such as paying bills and buying food.

In order to have the agent give more focus to more immediate rewards, but without losing
sight of an action's long-term effects, each reward is given a weight, called the
**discount** $\gamma^{i-1}$. This discount increases as the considered reward gets
farther into the future. The return is, therefore, the sum of the future discounted
rewards (going n states into the future):

G_n = \sum_i^n \gamma^{i-1}R_i = R0 + gamma * R1 + gamma^2 * R2 + ... + gamma^{n-1} * Rn
    =  R0 + gamma * (R1 + gamma * R2 + ... + gamma^{n-2}Rn)
    = R0 + gamma * G_{n+1}

The measure of how good an action is in a particular state, then, is determined by how
good the trajectory that it yields is. In other words, the Q-value for a given
state-action pair is the future discounted reward obtained by choosing that action
in that state and then playing optimally (so as to get the best trajectory):

Q(s,a) = R(s,a) + gamma * max_{a'}(Q(s',a'))

# Updating Q-values

Now that we know how the Q-values are determined, we turn to the next problem. Initially,
all of the Q-values in the Q-table are bogus since the agent doesn't know anything about
playing the game. So how are they updated so that they are actually informative?

This is done by having the agent interact with the environment and choose its actions
randomly. This process builds up a sample of trajectories that can be used to update
the Q-values because the agent can see which action choices in each state yield good
trajectories. The process works like this: get a trajectory -> update table -> get a
trajectory -> update table, etc.

Looking at the first trajectory T1, we see it's comprised of:

T1 = [(s1, a1), (s2, a2), ..., (sn, an)]

Therefore, starting at (s1, a1), we have two Q-values. The first is the initial bogus
guess that we put in the table and the second is the value we get from T1. We'll call
these Q_guess and Q_target. The difference between Q_guess and Q_target gives us an
error, called the **temporal-difference error** (TD error). It's obtained by
subtracting Q(s,a) from both sides of the above Q-value equation:

TD =  R(s,a) + gamma * max_{a'}(Q(s', a')) - Q(s,a) = Q_target - Q_guess

Now, here's where things get fun, conceptually. We know that neither of our two Q-values
are correct. This is because Q_guess was, well, a guess, and Q_target was obtained from
a bunch of other Q-value guesses along the trajectory. As such, when we're updating our
Q_guess, we want to scale the error.

Q_guess(s,a) = Q_guess(s,a) + alpha * TD,

where alpha is a scale factor called the **learning rate**. If alpha = 0, then there is
no update to the Q-value. If alpha = 1, then we update all the way to Q_target. As such,
alpha controls the size of the incremental update made to Q_guess.

Once this is done, we move on to (s2, a2) and do the same thing.

With a large sample-size of trajectories, and updating the Q-values as we go,we begin
to get a clearer picture of which trajectories are good and which are bad.

In other words, by sampling from reality and trying to match what we see, the Q-table
begins to converge, with the difference between Q_guess and Q_target decreasing over
time. This allows the agent to begin to **exploit** the knowledge it has in the Q-table
in order to get the highest score possible.

# Deep Q Networks

Q-tables are great, but for any environment with either a large or continuous state space,
they become computationally unwieldly. However, a Q-table is really just a tabulated
version of a function that maps states to actions. Conveniently, neural networks are
universal function approximators. This means that, instead of using a Q-table to tabulate
the unknown Q-function, we can use a neural network to approximate it.

The idea is exactly the same as above, but instead of updating Q-values, we update the
weights in the network (so that the output Q-values that are predicted for each action
for the given input state are updated).

The TD error now becomes the quantity that gets minimized in the network's loss function.
