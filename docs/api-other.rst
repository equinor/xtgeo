Other APIs
----------

Common utilities
^^^^^^^^^^^^^^^^

The :mod:`xtgeo.common` module provides shared utilities. Its
underscore-prefixed enums are re-exported for internal use across XTGeo
modules only, and are therefore not documented here.

.. autofunction:: xtgeo.common.null_logger

XTGDescription
""""""""""""""

.. autoclass:: xtgeo.common.XTGDescription

    .. autoclasstoc::

XTGShowProgress
"""""""""""""""

.. autoclass:: xtgeo.common.XTGShowProgress

    .. autoclasstoc::

Utilities
^^^^^^^^^

.. autofunction:: xtgeo.generic_hash

Roxar utilities
^^^^^^^^^^^^^^^

RoxUtils
""""""""

.. autoclass:: xtgeo.RoxUtils

    .. autoclasstoc::

Metadata (experimental)
^^^^^^^^^^^^^^^^^^^^^^^

MetadataRegularSurface
""""""""""""""""""""""

.. autoclass:: xtgeo.MetaDataRegularSurface

    .. autoclasstoc::

MetaDataRegularCube
"""""""""""""""""""

.. autoclass:: xtgeo.MetaDataRegularCube

    .. autoclasstoc::

MetaDataCPGeometry
""""""""""""""""""

.. autoclass:: xtgeo.MetaDataCPGeometry

    .. autoclasstoc::

MetaDataCPProperty
""""""""""""""""""

.. autoclass:: xtgeo.MetaDataCPProperty

    .. autoclasstoc::

MetaDataWell
""""""""""""

.. autoclass:: xtgeo.MetaDataWell

    .. autoclasstoc::

MetaDataTriangulatedSurface
"""""""""""""""""""""""""""

.. autoclass:: xtgeo.MetaDataTriangulatedSurface

    .. autoclasstoc::

Constants
^^^^^^^^^

XTGeo uses the following values to represent undefined (masked) nodes.

.. autodata:: xtgeo.UNDEF

.. autodata:: xtgeo.UNDEF_LIMIT

.. autodata:: xtgeo.UNDEF_INT

.. autodata:: xtgeo.UNDEF_INT_LIMIT

Exceptions
^^^^^^^^^^

.. autoexception:: xtgeo.BlockedWellsNotFoundError

.. autoexception:: xtgeo.DateNotFoundError

.. autoexception:: xtgeo.GridNotFoundError

.. autoexception:: xtgeo.InvalidFileFormatError

.. autoexception:: xtgeo.KeywordFoundNoDateError

.. autoexception:: xtgeo.KeywordNotFoundError

.. autoexception:: xtgeo.WellNotFoundError

.. autoexception:: xtgeo.XTGeoCLibError

Messaging
^^^^^^^^^

XTGeoDialog
"""""""""""

.. autoclass:: xtgeo.XTGeoDialog

    .. autoclasstoc::
