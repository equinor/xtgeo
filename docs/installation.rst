.. highlight:: shell

============
Installation
============


Stable release
--------------

The stable release is on /project/res, so it can be run as e.g.:

 from xtgeo.surface import RegularSurface


From sources
------------

The sources for XTGeo can be downloaded from the `Statoil Git repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://git.statil.no/xtgeo/pyxtgeo

Also you will need test data:

.. code-block:: console

   $ git clone git://git.statil.no/xtgeo/xtgeo-testdata

And you may perhaps need:

.. code-block:: console

    $ git clone git://git.statil.no/xtgeo/cxtgeo

Once you have a copy of the source, and you have a `virtual environment`_,
you can install it with:

.. code-block:: console

    $ make install


.. _Statoil Git repo: https://git.statoil.no/xtgeo/pyxtgeo
.. _virtual environment: http://docs.python-guide.org/en/latest/dev/virtualenvs/
