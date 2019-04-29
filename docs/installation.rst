.. highlight:: shell

============
Installation
============


Stable release
--------------

Within Equinor, the stable release is on /project/res or Komodo, so it can
be run as e.g.:

 import xtgeo


From sources
------------

The sources for XTGeo can be downloaded from the `Equinor Github repo`_.
Send a message to JRIV@equinor.com if your don't have access.

You can either clone the public repository:

.. code-block:: console

    $ git clone git@github.com:equinor/xtgeo

Also you will need test data at the same folder level as the source:

.. code-block:: console

   $ git clone git@github.com:equinor/xtgeo-testdata

Once you have a copy of the source, and you have a `virtual environment`_,
then always run tests (run first compile with make cc):

.. code-block:: console

   $ make cc
   $ make test

Next you can install it with:

.. code-block:: console

   $ make install

Or to install in developing mode with the VE:

.. code-block:: console

   $ make develop


.. _Equinor Github repo: https://github.com/equinor/xtgeo
.. _virtual environment: http://docs.python-guide.org/en/latest/dev/virtualenvs/
