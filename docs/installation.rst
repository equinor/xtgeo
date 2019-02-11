.. highlight:: shell

============
Installation
============


Stable release
--------------

The stable release is on /project/res or Komodo, so it can be run as e.g.:

 import xtgeo


From sources
------------

The sources for XTGeo can be downloaded from the `Equinor Github repo`_.
Send a message to JRIV@equinor.com if your don't have access.

You can either clone the public repository:

.. code-block:: console

    $ git clone git@github.com:equinor/xtgeo

Also you will need test data:

.. code-block:: console

   $ git clone git@github.com:equinor/xtgeo-testdata

Once you have a copy of the source, and you have a `virtual environment`_,
you can install it with:

.. code-block:: console

    $ make install


.. _Equinor Github repo: https://github.com/equinor/xtgeo
.. _virtual environment: http://docs.python-guide.org/en/latest/dev/virtualenvs/
