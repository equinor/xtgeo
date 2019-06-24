# -*- coding: utf-8 -*-

"""The surfaces module, which has the Surfaces class (collection of *Surface objects)"""
from __future__ import division, absolute_import
from __future__ import print_function

import xtgeo
from . import _surfs_import

xtg = xtgeo.common.XTGeoDialog()
logger = xtg.functionlogger(__name__)


class Surfaces(object):
    """Class for a collection of Surface objects, for operations that involves
    a number of surfaces.

    A collection of surfaces can be different things:

    * A list if surfaces in stratigraphic order
    * A collection of different realisations of the same surface
    * A collection of isochores

    .. seealso::
       Class :class:`~xtgeo.surface.regular_surface.RegularSurface` class.

    .. versionadded: 2.1.0
    """

    def __init__(self):

        self._surfaces = []  # list of RegularSurface objects
        self._subtype = None  # could be "tops", "isochores" or None
        self._order = None  # could be "same", "stratigraphic" or None

        # if args:
        #     # make instance from file import
        #     wfiles = args[0]
        #     fformat = kwargs.get("fformat", "rms_ascii")
        #     mdlogname = kwargs.get("mdlogname", None)
        #     zonelogname = kwargs.get("zonelogname", None)
        #     strict = kwargs.get("strict", True)
        #     self.from_files(
        #         wfiles,
        #         fformat=fformat,
        #         mdlogname=mdlogname,
        #         zonelogname=zonelogname,
        #         strict=strict,
        #         append=False,
        #     )

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

    # def get_statistics(self):
    #     "Returns a dictionary with statistical measures"
    #     pass

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
