.. _quickstart:

==========
Quickstart
==========

Installation
============

ANNA can be installed via :program:`pip`:

.. code-block:: bash

    pip install anna

For more, see :ref:`installation`.

Run Setup
=========

It is strongly recommended that each ANNA run is kept in its own directory
tree. The recommended structure is:

::

    runName
        |-- parameter_file.yaml
        |-- run_anna.py
        |-- output_directory/

If you'd like, rather than copy the :file:`run_anna.py` file to each run
directory, you can simply create a symlink to the original file:

.. code-block:: bash

    ln -s /path/to/original/run_anna.py /path/to/runName/run_anna.py

This directory structure keeps each run isolated from every other run, which
makes keeping things organized much easier.

.. note::

    The output directory does not have to exist prior to starting the run, as
    ANNA will take care of creating it for you. However, if it does exist,
    and you're starting a training run from the beginning, the output directory
    must be empty of files.

The last step in setting up a run is to properly set the values of the
parameters in the parameter file.

Usage
=====

ANNA is started via:

.. code-block:: bash

    python ./run_anna.py ./parameter_file.yaml [options]

ANNA can be easily stopped at any time by simply creating a file called
:file:`stop` in the output directory:

.. code-block:: bash

    touch /path/to/runName/output_directory/stop

When this file is detected, ANNA will finish its current step, save a
checkpoint file containing the agent's state, and then delete the stop file.
