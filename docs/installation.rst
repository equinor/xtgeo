.. highlight:: shell

============
Installation
============

From pip
--------

For a selection of platforms (currently Linux only) and Python versions:

 pip install xtgeo


Stable release in Equinor
-------------------------

Within Equinor, the stable release is pre-installed, so all you have
to do is:

.. code-block:: python

   import xtgeo



From sources
------------

This is only verified on Linux. You will need `swig` (version 2 or later)
installed, in addition to a C compiler (gcc)

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

Once you have a copy of the source, and you have a `virtual environment`_,
then always run tests (run first compile with make cc):

.. code-block:: console

   $ make test

Next you can install it with:

.. code-block:: console

   $ make install

Or to install in developing mode with the VE:

.. code-block:: console

   $ make develop


.. _Equinor Github repo: https://github.com/equinor/xtgeo
.. _virtual environment: http://docs.python-guide.org/en/latest/dev/virtualenvs/
