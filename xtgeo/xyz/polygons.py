"""XTGeo Polygons class"""

# #############################################################################
#
# NAME:
#    polygons.py
#
# AUTHOR(S):
#    Jan C. Rivenaes
#
# DESCRIPTION:
#    Polygons (connected points), which is subset of Points
#    Stored as Pandas in Python.
# TODO/ISSUES/BUGS:
#
# LICENCE:
#    Statoil property
# #############################################################################

from __future__ import print_function
import logging
import pandas as pd

from .points import Points


class Polygons(Points):
    """
    Class for a points set in the XTGeo framework.
    """

    def __init__(self, *args, **kwargs):

        """The __init__ (constructor) method.

        The instance can be made either from file or by a spesification::

        >>> xp = Polygons()
        >>> xp.from_file('somefilename', fformat='xyz')

        Args:
            xxx (nn): to come


        """

        clsname = "{}.{}".format(type(self).__module__, type(self).__name__)
        self.logger = logging.getLogger(clsname)
        self.logger.addHandler(logging.NullHandler())

        Points.__init__(self)

        self.logger.debug('Ran __init__ method')

    def from_wells(self, wells, zone, resample=1):

        """Get line segments from a list of wells and a zone number

        Args:
            wells (list): List of XTGeo well objects
            zone (int): Which zone to apply
            resample (int): If given, resample every N'th sample to make
                polylines smaller in terms of bit and bytes.
                1 = No resampling.


        Returns:
            None if well list is empty; otherwise the number of wells that
            have one or more line segments to return

        Raises:
            Todo
        """

        if len(wells) == 0:
            return None

        dflist = []
        for well in wells:
            wp = well.get_zone_interval(zone, resample=resample)
            if wp is not None:
                dflist.append(wp)

        if len(dflist) > 0:
            self._df = pd.concat(dflist, ignore_index=True)
            self._df.reset_index(inplace=True, drop=True)
        else:
            return None

        return len(dflist)
