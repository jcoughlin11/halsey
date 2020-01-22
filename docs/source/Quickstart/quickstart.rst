.. _quickstart:

==========
Quickstart
==========

Installation
============

Halsey can be installed via :program:`pip`:

.. code-block:: bash

    pip install halsey

For more, see :ref:`installation`.

Run Setup
=========

It is strongly recommended that each Halsey run is kept in its own directory
tree. The recommended structure is:

::

    runName
        |-- parameter_file.yaml
        |-- output_directory/

This directory structure keeps each run isolated from every other run, which
makes keeping things organized much easier.

.. note::

    The output directory does not have to exist prior to starting the run, as
    Halsey will take care of creating it for you. However, if it does exist,
    and you're starting a training run from the beginning, the output directory
    must be empty of files.

The last step in setting up a run is to properly set the values of the
parameters in the parameter file.

Usage
=====

Halsey is started via:

.. code-block:: bash

    halsey ./parameter_file.yaml [options]

Halsey can be easily stopped at any time by simply creating a file called
:file:`stop` in the output directory:

.. code-block:: bash

    touch /path/to/runName/output_directory/stop

When this file is detected, Halsey will finish its current step, save a
checkpoint file containing the agent's state, and then delete the stop file.
