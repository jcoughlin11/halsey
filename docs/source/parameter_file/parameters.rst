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
    * memory
    * action
    * navigation
    * frame

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

    * | **envName** : This is the name of the gym environment to train the agent
      | on, e.g., `SpaceInavders-v4`.

    * | **train** : A boolean flag. If set to `True` then the agent executes the
      | selected training loop.

    * | **test** : A boolean flag. If set to `True` then the agent actually plays
      | the game in order to have its performace evaluated.

    * | **timeLimit** : The maximum amount of time (in seconds) to let the agent
      | run for. This is primarily for use on clusters where it is common
      | practice to enforce a maximum run time for a given job. This allows for
      | the training progress to be saved and continued at a later date before
      | the system kills the job and lots of progress is potentially lost,
      | depending on how close the job termination was to the last natually
      | occurring checkpoint.

.. _ioparams:

io
==

The parameters in the io section are related to saving and loading checkpoint
files.

    * | **outputDir** : The path to the desired directory where the output will be
      | stored.
    * **fileBase** : The prefix that is prepended to each checkpoint file.

.. _trainingparams:

training
========

The parameters in this section control the training loop.

    * General

        - **nEpisodes** : The number of episodes used to train the agent.
        - | **maxEpisodeSteps** : The maximum number of steps allowed per episode.
          | This helps to prevent overly long episodes once the agent starts to get
          | quite good at the game.
        - | **batchSize** : The number of samples to use during each pass through
          | the network.
        - **savePeriod** : The number of episodes between saving checkpoints.
    * Mode

        - **qTrainer** : A traditional deep-Q learning training loop.

.. _brainparams:

brain
=====

The parameters in this section are related to the neural network(s).
