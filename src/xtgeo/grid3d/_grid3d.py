# -*- coding: utf-8 -*-
"""Private baseclass for Grid and GridProperties, not to be used by itself."""

from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()
logger = xtg.functionlogger(__name__)


class _Grid3D(object):
    """Abstract base class for Grid3D."""

    def __init__(self):

        self._ncol = 4
        self._nrow = 3
        self._nlay = 5

    @property
    def ncol(self) -> int:
        """Returns the NCOL (NX or Ncolumns) number of cells."""
        return self._ncol

    @property
    def nrow(self) -> int:
        """Returns the NROW (NY or Nrows) number of cells."""
        return self._nrow

    @property
    def nlay(self) -> int:
        """Returns the NLAY (NZ or Nlayers) number of cells."""
        return self._nlay

    def _evaluate_mask(self, mask) -> bool:
        xtg.warn(
            "Use of keyword 'mask' in argument list is deprecated, "
            "use alternative specified in API instead! In: {}".format(self)
        )

        if not isinstance(mask, bool):
            raise ValueError('Wrong value or use of keyword "mask"')

        if mask is False:
            return False

        return True
