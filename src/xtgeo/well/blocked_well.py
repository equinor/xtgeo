# -*- coding: utf-8 -*-
"""XTGeo blockedwell module"""

from __future__ import print_function, absolute_import

import xtgeo
from .well1 import Well
from . import _blockedwell_roxapi

xtg = xtgeo.common.XTGeoDialog()
logger = xtg.functionlogger(__name__)


# =============================================================================
# METHODS as wrappers to class init + import


def blockedwell_from_file(
    bwfile, fformat="rms_ascii", mdlogname=None, zonelogname=None, strict=False
):
    """Make an instance of a BlockedWell directly from file import.

    Args:
        bmfile (str): Name of file
        fformat (str): See :meth:`Well.from_file`
        mdlogname (str): See :meth:`Well.from_file`
        zonelogname (str): See :meth:`Well.from_file`
        strict (bool): See :meth:`Well.from_file`

    Example::

        import xtgeo
        mybwell = xtgeo.blockedwell_from_file('somewell.xxx')
    """

    obj = BlockedWell()

    obj.from_file(
        bwfile,
        fformat=fformat,
        mdlogname=mdlogname,
        zonelogname=zonelogname,
        strict=strict,
    )

    return obj


def blockedwell_from_roxar(project, gname, bwname, wname, lognames=None, ijk=True):

    """This makes an instance of a BlockedWell directly from Roxar RMS.

    For arguments, see :meth:`BlockedWell.from_roxar`.

    Example::

        # inside RMS:
        import xtgeo
        mylogs = ['ZONELOG', 'GR', 'Facies']
        mybw = xtgeo.blockedwell_from_roxar(project, 'Simgrid', 'BW', '31_3-1',
                                            lognames=mylogs)

    """

    obj = BlockedWell()

    obj.from_roxar(project, gname, bwname, wname, ijk=ijk, lognames=lognames)

    return obj


# =============================================================================
# CLASS


class BlockedWell(Well):
    """Class for a blocked well in the XTGeo framework, subclassed from the
    Well class.

    Similar to Wells, the blocked well logs are stored as Pandas dataframe,
    which make manipulation easy and fast.

    The well trajectory are here represented as logs, and XYZ have magic names:
    X_UTME, Y_UTMN, Z_TVDSS, which are the three first Pandas columns.

    Other geometry logs has also 'semi-magic' names:

    M_MDEPTH or Q_MDEPTH: Measured depth, either real/true (M...) or
    quasi computed/estimated (Q...). The Quasi may be incorrect for
    all uses, but sufficient for some computations.

    Similar for M_INCL, Q_INCL, M_AZI, Q_ASI.

    I_INDEX, J_INDEX, K_INDEX: They are grid indices. For practical reasons
    they are treated as a CONT logs, since the min/max grid indices usually are
    unknown, and hence making a code index is not trivial.

    All Pandas values (yes, discrete also!) are stored as float64
    format, and undefined values are Nan. Integers are stored as Float due
    to the lacking support for 'Integer Nan' (currently lacking in Pandas,
    but may come in later Pandas versions).

    Note there is a method that can return a dataframe (copy) with Integer
    and Float columns, see :meth:`get_filled_dataframe`.

    The instance can be made either from file or (todo!) by spesification::

        >>> well1 = BlockedWell('somefilename')  # assume RMS ascii well
        >>> well2 = BlockedWell('somefilename', fformat='rms_ascii')
        >>> well3 = xtgeo.blockedwell_from_file('somefilename')

    For arguments, see method under :meth:`from_file`.

    """

    VALID_LOGTYPES = {"DISC", "CONT"}

    def __init__(self, *args, **kwargs):

        super(BlockedWell, self).__init__(*args, **kwargs)

        self._gridname = None

    @property
    def gridname(self):
        """ Returns or set (rename) the grid name that the blocked wells
        belongs to."""
        return self._gridname

    @gridname.setter
    def gridname(self, newname):
        self._gridname = newname

    def copy(self):

        newbw = super(BlockedWell, self).copy()

        newbw._gridname = self._gridname

    def from_roxar(self, *args, **kwargs):
        """Import (retrieve) a single blocked well from roxar project.

        Note this method works only when inside RMS, or when RMS license is
        activated.

        Args:
            project (str): Magic string 'project' or file path to project
            gname (str): Name of GridModel icon in RMS
            bwname (str): Name of Blocked Well icon in RMS, usually 'BW'
            wname (str): Name of well, as shown in RMS.
            lognames (list): List of lognames to include, or use 'all' for
                all current blocked logs for this well.
            realisation (int): Realisation index (0 is default)
            ijk (bool): If True, then logs with grid IJK as I_INDEX, etc
        """
        project = args[0]
        gname = args[1]
        bwname = args[2]
        wname = args[3]
        lognames = kwargs.get("lognames", None)
        ijk = kwargs.get("ijk", False)
        realisation = kwargs.get("realisation", 0)

        _blockedwell_roxapi.import_bwell_roxapi(
            self,
            project,
            gname,
            bwname,
            wname,
            lognames=lognames,
            ijk=ijk,
            realisation=realisation,
        )

        self._ensure_consistency()

    def to_roxar(self, *args, **kwargs):
        """Set (export) a single blocked well item inside roxar project.

        Note this method works only when inside RMS, or when RMS license is
        activated.

        Args:
            project (str): Magic string 'project' or file path to project
            gname (str): Name of GridModel icon in RMS
            bwname (str): Name of Blocked Well icon in RMS, usually 'BW'
            wname (str): Name of well, as shown in RMS.
            realisation (int): Realisation index (0 is default)
        """
        project = args[0]
        gname = args[1]
        bwname = args[2]
        wname = args[3]
        realisation = kwargs.get("realisation", 0)

        _blockedwell_roxapi.export_bwell_roxapi(
            self, project, gname, bwname, wname, realisation=realisation,
        )
