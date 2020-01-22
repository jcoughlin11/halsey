.. _installation:

============
Installation
============

It is strongly recommended that you install Halsey in a `virtual environment <https://virtualenv.pypa.io/en/latest/>`_.
If you need to manage multiple python versions, then `pyenv <https://github.com/pyenv/pyenv>`_
is an excellent tool for doing so.

Halsey has several dependencies:

    * `python <https://www.python.org/>`_ (3.6+)
    * `numpy <https://numpy.org/>`_ (1.17+)
    * `pyyaml <https://pyyaml.org/>`_ (5.1+)
    * `h5py <https://www.h5py.org/>`_ (2.10+)
    * `gym <https://gym.openai.com/>`_ (0.15.4+)
    * `atari_py <https://github.com/openai/atari-py/tree/master/atari_py>`_ (0.2.6+)
    * `cython <https://cython.org/>`_ (0.29.14+)
    * `scikit-image <https://scikit-image.org/>`_ (0.16.2+)
    * `tk <https://docs.python.org/3/library/tk.html>`_ (0.1.0+)
    * `scipy <https://www.scipy.org/>`_ (1.3+)
    * `imread <https://matplotlib.org/api/_as_gen/matplotlib.pyplot.imread.html>`_ (0.7.1+)
    * `tensorflow <https://www.tensorflow.org/>`_ (2.1+)
    * `sphinx <http://www.sphinx-doc.org/en/master/index.html>`_ (2.3.1+). **Needed for building docs only.**
    * `sphinx_rtd_theme <https://sphinx-rtd-theme.readthedocs.io/en/stable/>`_ (0.4.3+) **Needed for building docs only.**

Don't worry, though, the standard installation of Halsey takes care of all of that for you!
This project makes use of `poetry <https://python-poetry.org/docs/>`_ to manage dependencies.

Install Using Pip
=================
The easiest way to install Halsey is via :program:`pip`.

.. code-block:: bash

    pip install halsey

This will install Halsey and all of its dependencies.


Installing From Source
======================
You can also install Halsey from source, like so:

.. code-block:: bash

    git clone https://github.com/jcoughlin11/halsey
    cd halsey/
    pip install .


Building The Documentation
==========================
If you'd like, you can build this documentation in `html` format. This is
done, starting from the Halsey repository, via:

.. code-block:: bash

    cd docs/
    make html

The `html` documentation can then be found in :file:`docs/build/html`. The
`html` docs are viewable with any modern browser, *e.g.*

.. code-block:: bash

    firefox docs/build/html/index.html
