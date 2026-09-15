Array handling
==============

.. _asmasked-vs-activeonly:

``asmasked`` and ``activeonly``
--------------------------------

Several methods use the ``asmasked`` or ``activeonly`` argument to control how
inactive or undefined cells and nodes are returned. These arguments are
different names for the same conceptual flag when passed to different methods:
whether inactive or undefined entries are included. They do not, however,
produce exactly the same output:

* ``asmasked=True`` preserves the shape of the data and masks inactive or
  undefined entries. ``asmasked=False`` includes those entries, using the
  representation documented by the method. Methods that return an array
  directly commonly return a ``numpy.ma.MaskedArray`` when this argument is
  true and a ``numpy.ndarray`` when it is false.
* ``activeonly=True`` removes inactive or undefined entries from the result.
  ``activeonly=False`` includes them, using the representation documented by
  the method, commonly ``None``, ``numpy.nan``, or a specified fill value.

In other words, both arguments control the treatment of inactive or undefined
entries. Use ``asmasked`` when a method offers a choice between a masked and an
unmasked array, and ``activeonly`` when a method offers a choice between
filtering and retaining those entries.