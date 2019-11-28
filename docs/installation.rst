.. highlight:: shell

============
Installation
============

From pip
--------

For a selection of platforms (Linux and Windows) and Python versions:

.. code-block:: console

   $ pip install xtgeo

For Windows, a `manual install of Shapely`_ is currently required as a first step.


Stable release in Equinor
-------------------------

Within Equinor, the stable release is pre-installed, so all you have
to do is:

.. code-block:: python

   import xtgeo


From github
------------

You will need `swig` (version 3 or later) installed, in addition to a C compiler (see below).

.. code-block:: console

   $ pip install git+https://github.com/equinor/xtgeo


From downloaded sources
-----------------------

You will need `swig`_ (version 3 or later) installed, in addition to a C compiler.
Tested compilers are:

* gcc on Linux (Version 4 and later)
* Visual studio 2015 and 2017 on Windows

The sources for XTGeo can be downloaded from the `Equinor Github repo`_.

You can either clone the public repository:

.. code-block:: console

   $ git clone git@github.com:equinor/xtgeo

Also you will need test data at the same folder level as the source:

.. code-block:: console

   $ git clone git@github.com:equinor/xtgeo-testdata

Hence folder structure may look like

.. code-block:: console

   /some/path/to/xtgeo
   /some/path/to/xtgeo-testdata

For required python packages, see the requirements*.txt files and the
pyproject.toml file in the root folder.

Once you have a copy of the source, and you have a `virtual environment`_,
then always run tests (run first compile and install with ``pip install .``):

.. code-block:: console

   $ pytest

Next you can install it with:

.. code-block:: console

   $ pip install .

Or to install in developing mode with the VE:

.. code-block:: console

   $ pip install -e



.. _Equinor Github repo: https://github.com/equinor/xtgeo
.. _virtual environment: http://docs.python-guide.org/en/latest/dev/virtualenvs/
.. _manual install of Shapely: https://towardsdatascience.com/install-shapely-on-windows-72b6581bb46c
.. _swig: http://swig.org
