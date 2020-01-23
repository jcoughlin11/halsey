.. _parameterfile:

==================
The Parameter File
==================

Halsey is controlled via a parameter file. This parameter file is written in
the `yaml <https://en.wikipedia.org/wiki/YAML>`_ format, which makes it both
human-readable and easy to parse with python.

An example parameter file can be found `here <https://github.com/jcoughlin11/halsey/blob/master/params.yaml>`_

The parameter file is structured into **sections**. Each section is related to
one of the main aspects of the code. The default sections are:

    * :ref:`runparams`
    * :ref:`ioparams`
    * :ref:`trainingparams`
    * :ref:`brainparams`
    * :ref:`memoryparams`
    * :ref:`actionparams`
    * :ref:`navigationparams`
    * :ref:`frameparams`

Each section will be covered in detal below. Further, each section in the
parameter file is broken up into one, potentially two, **subsections** called
**general** and **mode**.

The general subsection contains those parameters that are common to all options
within the section. For example, in the **memory** section, every type of
memory buffer used in reinforcement learning requires a parameter describing its
maximum size; as such, **maxSize** is listed under the general subsection of the
memory section.

.. code-block:: yaml

    memory:
        general:
            maxSize : 100

The mode subsection contains **options** for a given section, along with the
parameters specific to each option. Continuing with the example from above,
for each run it is necessary to choose a type of memory buffer, such as an
experience memory buffer or an episodic memory buffer. As such, **only one** of
these options can be selected (otherwise an error will be thrown). Options are
turned on and off by setting their **enabled** attribute to either `True` or
`False`.

.. code-block:: yaml

    memory:
        mode:
            experience:
                enabled : True
            episode:
                enabled : False

The reasoning behind structuring the parameter file in this way as opposed to
simple having a `mode : <option name>` construction like the parameters in the
general subsection do was threefold. First, this construction eliminates the need
for extra parameter parsing in order to remove unused parameters from the
object used to hold them in the code. Second, it eliminates the need to
carry around unused parameters in the parameter container object if the first
option were not taken, and third, it is modular, easy to read, and easy to use.

Adding addtional options is as simple as adding a new block to the mode
subsection.

.. note::

    The names of the variables in the parameter file are also the names used in
    the code for those parameters.

.. _runparams:

run
===

The parameters in the run section are directly related to controlling the run
at the highest level.

    * **envName** : This is the name of the gym environment to train the agent on, e.g., `SpaceInavders-v4`.

    * **train** : A boolean flag. If set to `True` then the agent executes the selected training loop.

    * **test** : A boolean flag. If set to `True` then the agent actually plays the game in order to have its performace evaluated.

    * **timeLimit** : The maximum amount of time (in seconds) to let the agent run for. This is primarily for use on clusters where it is common practice to enforce a maximum run time for a given job. This allows for the training progress to be saved and continued at a later date before the system kills the job and lots of progress is potentially lost, depending on how close the job termination was to the last natually occurring checkpoint.

.. _ioparams:

io
==

The parameters in the io section are related to saving and loading checkpoint
files.

    * **outputDir** : The path to the desired directory where the output will be stored.
    * **fileBase** : The prefix that is prepended to each checkpoint file.

.. _trainingparams:

training
========

The parameters in this section control the training loop.

    * General

        - **nEpisodes** : The number of episodes used to train the agent.
        - **maxEpisodeSteps** : The maximum number of steps allowed per episode. This helps to prevent overly long episodes once the agent starts to get quite good at the game.
        - **batchSize** : The number of samples to use during each pass through the network.
        - **savePeriod** : The number of episodes between saving checkpoints.
    * Mode

        - **qTrainer** : A traditional deep-Q learning training loop.

.. _brainparams:

brain
=====

The parameters in this section are related to the neural network(s).

    * General

        - **architecture** : The name of the neural network architecture to use.
        - **discount** : Describes how much importance the agent gives to future rewards as opposed to immediate rewards, with larger values giving more importance to future rewards.
        - **learningRate** : Describes how drastically the network weights are updated at each time step, with a value of zero resulting in no update and a value of 1 resulting in a very drastic update. It acts as a step size in the Bellman equation.
        - **loss** : The name of the function to be minimized during the learning process.
        - **optimizer** : The name of the method used for minimizing the loss function.
    * Mode

        - **vanillaQ** : The learning algorithm described in [Minh13]_.
        - **doubleDqn** : The learning algorithm described in [Hasselt15]_.
        - **fixedQ** : The learning algorithm described in [Lillicrap15]_.

            + **fixedQSteps** : The number of time steps that must elapse before updating the weights of the target network with those of the primary network.

.. _memoryparams:

memory
======

The parameters in this section are related to the memory buffer.

    * General

        - **maxSize** : The max number of samples allowed to be held in the memory buffer. Once this limit is reached, the oldest samples begin to be deleted from the buffer in order to make room for the newest samples.
        - **pretrainLen** : The number of samples to initially fill the memory buffer with in order to avoid the empty memory problem at the start of training.
    * Mode

        - **experience** : Selecting this type of memory causes the memory buffer to fill with individual (state, action, reward, next state, done) experiences.
        - **episode** : Selecting this type of memory causes the memory buffer to fill with entire episodes, traces of which are selected for use in learning.

.. _actionparams:

action
======

The parameters in this section are related to the issue of exploration vs.
exploitation during the training process.

    * Mode

        - **epsilonGreedy** : This strategy chooses a random action with probability :math:`\epsilon` and expoints the agent's knowledge of the game otherwise. At the start of training we have :math:`\epsilon \approx 1` and, as training continues we have :math:`\lim_{t\rightarrow\infty} \epsilon \rightarrow \epsilon_f`; that is, the probability of selecting a random action approaches some lower limit (usually non-zero in order to always allow for a chance of trying something new).
        - **epsDecayRate** : How quickly :math:`\epsilon` anneals from its initial value to its final value.
        - **epsilonStart** : The initial value of :math:`\epsilon`.
        - **epsilonStop** : The lowest :math:`\epsilon` is allowed to get.

.. _navigationparams:

navigation
==========

The parameters in this section are related to how the agent steps through data
provided by the game environment (e.g., Markovian vs. partially-observable Markovian).

    * Mode

        - **frameByFrame** : Standard Markovian processing. Goes through the game seeing every frame in order.

.. _frameparams:

frame
=====

The parameters in this section are related to how the agent preprocesses the game frames
that it receives (e.g., cropping, scaling, etc.). The crop parameters are useful for
removing unnecessary parts of the game screen. The shrink parameters are used for scaling
the image (usually down to a smaller, more manageable size).

    * Mode

        - **vanilla** : The standard image processing pipeline from [Minh13]_.

            + **cropBot** : The number of rows to remove from the image starting with the bottom.
            + **cropLeft** : The number of columns to remove from the image starting from the left.
            + **cropRight** : The number of columns to remove from the image starting from the right.
            + **cropTop** : The number of rows to remove from the image starting from the top.
            + **shrinkCols** : The number of columns to use in the scaled version of the image.
            + **srhinkRows** : The number of rows to use in the scaled version of the image.
            + **traceLen** : The number of frames to stack together as one state. This helps solve the problem of temporal limitation.

.. [Minh13] `Minh, V., **et al**., "Playing Atari with Deep Reinforcement Learning,"
    CoRR, vol. 1312, 2013. <https://arxiv.org/abs/1312.5602>`_
.. [Hasselt15] `van Hasselt, H., **et al**., "Deep Reinforcement Learning with Double Q-Learning,"
    CoRR, vol. 1509, 2015. <https://arxiv.org/abs/1509.06461>`_
.. [Lillicrap15] `Lillicrap, T., **et al**., "Continuous Control with Deep Reinforcement Learning,"
    arXiv e-prints, 2015. <https://arxiv.org/abs/1509.02971>`_
