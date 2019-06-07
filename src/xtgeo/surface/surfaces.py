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

    See also the :class:`~xtgeo.surface.regular_surface.RegularSurface` class.
    """

    def __init__(self, *args, **kwargs):

        self._surfaces = []  # list of RegularSurface objects

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

        dsc.txt("Wells", self.names)

        if flush:
            dsc.flush()
            return None

        return dsc.astext()

    # def copy(self):
    #     """Copy a Wells instance to a new unique instance (a deep copy)."""

    #     new = Wells()

    #     for well in self._wells:
    #         newwell = well.copy()
    #         new._props.append(newwell)

    #     return new

    # def get_surface(self, name):
    #     """Get a RegularSurface() instance by name, or None"""

    #     logger.info("Asking for a surface with name %s", name)
    #     for surf in self._surfaces:
    #         if surf.name == name:
    #             return surf
    #     return None

    # def from_files(
    #     self,
    #     filelist,
    #     fformat="rms_ascii",
    #     mdlogname=None,
    #     zonelogname=None,
    #     strict=True,
    #     append=True,
    # ):

        # """Import wells from a list of files (filelist).

        # Args:
        #     filelist (list of str): List with file names
        #     fformat (str): File format, rms_ascii (rms well) is
        #         currently supported and default format.
        #     mdlogname (str): Name of measured depth log, if any
        #     zonelogname (str): Name of zonation log, if any
        #     strict (bool): If True, then import will fail if
        #         zonelogname or mdlogname are asked for but not present
        #         in wells.
        #     append (bool): If True, new wells will be added to existing
        #         wells.

        # Example:
        #     Here the from_file method is used to initiate the object
        #     directly::

        #     >>> mywells = Wells(['31_2-6.w', '31_2-7.w', '31_2-8.w'])
        # """

        # if not append:
        #     self._wells = []

        # # file checks are done within the Well() class
        # for wfile in filelist:
        #     try:
        #         wll = xtgeo.well.Well(
        #             wfile,
        #             fformat=fformat,
        #             mdlogname=mdlogname,
        #             zonelogname=zonelogname,
        #             strict=strict,
        #         )
        #         self._wells.append(wll)
        #     except ValueError as err:
        #         xtg.warn("SKIP this well: {}".format(err))
        #         continue
        # if not self._wells:
        #     xtg.warn("No wells imported!")
