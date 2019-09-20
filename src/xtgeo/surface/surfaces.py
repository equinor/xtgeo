# -*- coding: utf-8 -*-

"""The surfaces module, which has the Surfaces class (collection of surface objects)"""

from __future__ import division, absolute_import
from __future__ import print_function

import numpy as np

import xtgeo
from . import _surfs_import

xtg = xtgeo.common.XTGeoDialog()
logger = xtg.functionlogger(__name__)


class Surfaces(object):
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

    .. versionadded:: 2.1.0
    """

    def __init__(self, *args, **kwargs):

        self._surfaces = []  # list of RegularSurface objects
        self._subtype = None  # could be "tops", "isochores" or None
        self._order = None  # could be "same", "stratigraphic" or None

        if args:
            self.append(args[0])
            self._subtype = kwargs.get("subtype", None)
            self._order = kwargs.get("order", None)

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
                self._surfaces.append(item)
            else:
                try:
                    sobj = xtgeo.surface_from_file(item, fformat="guess")
                    self._surfaces.append(sobj)
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

    def from_grid3d(self, grid, subgrids=True, rfactor=1):
        """Derive surfaces from a 3D grid"""
        _surfs_import.from_grid3d(self, grid, subgrids, rfactor)

    def apply(self, func, *args, **kwargs):
        """Apply a function to the Surfaces array.

        The return value of the function (numpy nan comptatible) will be a
        numpy array of the same shape as the first surface.

        E.g. surfs.apply(np.nanmean, axis=0) will return the mean surface.

        Args:
            func: Function to apply
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

        template.values = func(xlist, *args, **kwargs)
        return template

    def statistics(self):
        """Return statistical measures from the surfaces.

        The statistics returned is:
        * mean: the arithmetic mean surface
        * std: the standard deviation surface (where ddof = 1)

        Currently this function expects that the surfaces all have the same
        shape/topology.

        Returns:
            dict: A dictionary of statistical measures, see list above

        Raises:
            ValueError: If surfaces differ in topology.

        Example::

            surfs = Surfaces(mylist)  # mylist is a collection of files
            stats = surfs.statistics()
            # export the mean surface
            stats["mean"].to_file("mymean.gri")
        """
        result = {}

        template = self.surfaces[0].copy()

        slist = []
        for surf in self.surfaces:
            status = template.compare_topology(surf, strict=False)
            if not status:
                raise ValueError("Cannot do statistics, surfaces differ in topology")
            slist.append(np.ma.filled(surf.values, fill_value=np.nan))

        xlist = np.array(slist)

        # mean
        template.values = np.nanmean(xlist, axis=0)
        result["mean"] = template.copy()

        # std, run with degree of freedom ddof=1, similar to RMS
        template.values = np.nanstd(xlist, axis=0, ddof=1)
        result["std"] = template.copy()

        # # median P50 (SLOW!)
        # template.values = np.nanpercentile(xlist, 50, axis=0)
        # result["median"] = template.copy()
        # result["p50"] = result["median"]

        # # P10
        # template.values = np.nanpercentile(xlist, 10, axis=0)
        # result["p10"] = template.copy()

        # # P90
        # template.values = np.nanpercentile(xlist, 90, axis=0)
        # result["p90"] = template.copy()

        return result
