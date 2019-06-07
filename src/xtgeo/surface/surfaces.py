# -*- coding: utf-8 -*-

"""The surfaces module, which has the Surfaces class (collection of *Surface objects)"""
from __future__ import division, absolute_import
from __future__ import print_function

import xtgeo

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

    def __init__(self, *args, **kwargs):

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

    # @property
    # def names(self):
    #     """Returns a list of well names (read only).

    #     Example::

    #         namelist = wells.names
    #         for prop in namelist:
    #             print ('Well name is {}'.format(name))

    #     """

    #     wlist = []
    #     for wel in self._wells:
    #         wlist.append(wel.name)

    #     return wlist

    # @property
    # def wells(self):
    #     """Returns or sets a list of XTGeo Well objects, None if empty."""
    #     if not self._wells:
    #         return None

    #     return self._wells

    # @wells.setter
    # def wells(self, well_list):

    #     for well in well_list:
    #         if not isinstance(well, xtgeo.well.Well):
    #             raise ValueError("Well in list not valid Well object")

    #     self._wells = well_list

    def describe(self, flush=True):
        """Describe an instance by printing to stdout"""

        dsc = xtgeo.common.XTGDescription()
        dsc.title("Description of {} instance".format(self.__class__.__name__))
        dsc.txt("Object ID", id(self))

        dsc.txt("Surfaces", self.names)

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

    def get_statistics(self):
        "Returns a dictionary with statistical measures"
        pass

    def get_surface(self, name):
        """Get a RegularSurface() instance by name, or return None if name not found"""

        logger.info("Asking for a surface with name %s", name)
        for surf in self._surfaces:
            if surf.name == name:
                return surf
        return None

    def from_grid3d(self, subgrids=True):
        """Derive surfaces from a 3D grid"""
        pass
