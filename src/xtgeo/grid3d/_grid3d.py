# -*- coding: utf-8 -*-
"""Private baseclass for Grid and GridProperties, not to be used by itself."""

from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()


class _Grid3D:
    """Abstract base class for Grid3D."""

    def __init__(self, ncol: int = 4, nrow: int = 3, nlay: int = 5):
        self._ncol = ncol
        self._nrow = nrow
        self._nlay = nlay

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

    def _evaluate_mask(self, mask: bool) -> bool:
        xtg.warn(
            f"Use of keyword 'mask' in argument list is deprecated, use alternative "
            f"specified in API instead! In: {self}"
        )

        if not isinstance(mask, bool):
            raise ValueError('Wrong value or use of keyword "mask"')

        return mask
