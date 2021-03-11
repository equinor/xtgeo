# -*- coding: utf-8 -*-

"""The surfaces module, which has the Surfaces class (collection of surface objects)."""
from xtgeo import RegularSurface

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from typing import Optional, List
import warnings
import deprecation
import numpy as np

import xtgeo
from . import _surfs_import

xtg = xtgeo.common.XTGeoDialog()
logger = xtg.functionlogger(__name__)


def surfaces_from_grid(grid, subgrids=True, rfactor=1):
    surf, subtype, order = _surfs_import.from_grid3d(grid, subgrids, rfactor)
    return Surfaces(surfaces=surf, subtype=subtype, order=order)


class Surfaces:
    """Class for a collection of Surface objects, for operations that involves
    a number of surfaces, such as statistical numbers.

    A collection of surfaces can be different things:

    * A list if surfaces in stratigraphic order
    * A collection of different realisations of the same surface
    * A collection of isochores

    Args:
        input (list, optional): A list of XTGeo objects and/or file names)
        subtype (str): "tops", "isochores", or None (default)
        order (str): Assummed order: "same", "stratigraphic", None(default)

    .. seealso::
       Class :class:`~xtgeo.surface.regular_surface.RegularSurface` class.

    .. versionadded:: 2.1
    """

    def __init__(
        self,
        surfaces: Optional[List[RegularSurface]] = None,
        subtype: Optional[Literal["tops", "isochores"]] = None,
        order: Optional[Literal["same", "stratigraphic"]] = None,
    ):
        self._surfaces = []
        if surfaces is not None:
            self.append(surfaces)
        self._subtype = subtype
        self._order = order

    @property
    def surfaces(self):
        """Get or set a list of individual surfaces"""
        return self._surfaces

    @surfaces.setter
    def surfaces(self, slist):
        if not isinstance(slist, list):
            raise ValueError("Input not a list")

        for elem in slist:
            if not isinstance(elem, xtgeo.RegularSurface):
                raise ValueError("Element in list not a valid type of Surface")

        self._surfaces = slist

    def append(self, slist):
        """Append surfaces from either a list of RegularSurface objects,
        a list of files, or a mix."""
        for item in slist:
            if isinstance(item, xtgeo.RegularSurface):
                self.surfaces.append(item)
            else:
                try:
                    sobj = xtgeo.surface_from_file(item, fformat="guess")
                    self.surfaces.append(sobj)
                except OSError:
                    xtg.warnuser("Cannot read as file, skip: {}".format(item))

    def describe(self, flush=True):
        """Describe an instance by printing to stdout"""

        dsc = xtgeo.common.XTGDescription()
        dsc.title("Description of {} instance".format(self.__class__.__name__))
        dsc.txt("Object ID", id(self))

        for inum, surf in enumerate(self.surfaces):
            dsc.txt("Surface:", inum, surf.name)

        if flush:
            dsc.flush()
            return None

        return dsc.astext()

    def copy(self):
        """Copy a Surfaces instance to a new unique instance (a deep copy)."""

        new = Surfaces()

        for surf in self._surfaces:
            newsurf = surf.copy()
            new._surfaces.append(newsurf)

        new._order = self._order
        new._subtype = self._subtype

        return new

    def get_surface(self, name):
        """Get a RegularSurface() instance by name, or return None if name not found"""

        logger.info("Asking for a surface with name %s", name)
        for surf in self._surfaces:
            if surf.name == name:
                return surf
        return None

    @deprecation.deprecated(
        deprecated_in="2.15",
        removed_in="4.0",
        current_version=xtgeo.version,
        details="Use xtgeo.surface.surfaces_from_grid() instead",
    )
    def from_grid3d(self, grid, subgrids=True, rfactor=1):
        """Derive surfaces from a 3D grid"""
        self.surfaces, self._subtype, self._order = _surfs_import.from_grid3d(
            grid, subgrids, rfactor
        )

    def apply(self, func, *args, **kwargs):
        """Apply a function to the Surfaces array.

        The return value of the function (numpy nan comptatible) will be a
        numpy array of the same shape as the first surface.

        E.g. surfs.apply(np.nanmean, axis=0) will return the mean surface.

        Args:
            func: Function to apply, e.g. np.nanmean
            args: The function arguments
            kwargs: The function keyword arguments

        Raises:
            ValueError: If surfaces differ in topology.

        """
        template = self.surfaces[0].copy()
        slist = []
        for surf in self.surfaces:
            status = template.compare_topology(surf, strict=False)
            if not status:
                raise ValueError("Cannot do statistics, surfaces differ in topology")
            slist.append(np.ma.filled(surf.values, fill_value=np.nan))

        xlist = np.array(slist)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
            template.values = func(xlist, *args, **kwargs)

        return template

    def statistics(self, percentiles=None):
        """Return statistical measures from the surfaces.

        The statistics returned is:
        * mean: the arithmetic mean surface
        * std: the standard deviation surface (where ddof = 1)
        * percentiles: on demand (such operations may be slow)

        Currently this function expects that the surfaces all have the same
        shape/topology.

        Args:
            percentiles (list of float): If defined, a list of perecentiles to evaluate
                e.g. [10, 50, 90] for p10, p50, p90

        Returns:
            dict: A dictionary of statistical measures, see list above

        Raises:
            ValueError: If surfaces differ in topology.

        Example::

            surfs = Surfaces(mylist)  # mylist is a collection of files
            stats = surfs.statistics()
            # export the mean surface
            stats["mean"].to_file("mymean.gri")

        .. versionchanged:: 2.13 Added `percentile`
        """
        result = {}

        template = self.surfaces[0].copy()

        slist = []
        for surf in self.surfaces:
            status = template.compare_topology(surf, strict=False)
            if not status:
                raise ValueError("Cannot do statistics, surfaces differ in topology")
            slist.append(np.ma.filled(surf.values, fill_value=np.nan).ravel())

        xlist = np.array(slist)

        template.values = np.ma.masked_invalid(xlist).mean(axis=0)
        result["mean"] = template.copy()
        template.values = np.ma.masked_invalid(xlist).std(axis=0, ddof=1)
        result["std"] = template.copy()

        if percentiles is not None:

            # nan on a axis tends to give warnings that are not a worry; suppress:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
                res = np.nanpercentile(xlist, percentiles, axis=0)

            for slice, prc in enumerate(percentiles):
                template.values = res[slice, :]
                result["p" + str(prc)] = template.copy()
                if prc == 50:
                    result["median"] = result["p50"]

        return result
