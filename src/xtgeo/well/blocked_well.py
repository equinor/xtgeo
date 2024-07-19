"""XTGeo blockedwell module"""

import pandas as pd

from xtgeo.common._xyz_enum import _AttrName
from xtgeo.common.log import null_logger

from . import _blockedwell_roxapi
from .well1 import Well

logger = null_logger(__name__)


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

        >>> import xtgeo
        >>> well3 = xtgeo.blockedwell_from_file(well_dir + '/OP_1.bw')
    """

    return BlockedWell._read_file(
        bwfile,
        fformat=fformat,
        mdlogname=mdlogname,
        zonelogname=zonelogname,
        strict=strict,
    )

    # return obj


def blockedwell_from_roxar(
    project, gname, bwname, wname, lognames=None, ijk=True, realisation=0
):
    """This makes an instance of a BlockedWell directly from Roxar RMS.

    For arguments, see :meth:`BlockedWell.from_roxar`.

    Example::

        # inside RMS:
        import xtgeo
        mylogs = ['ZONELOG', 'GR', 'Facies']
        mybw = xtgeo.blockedwell_from_roxar(project, 'Simgrid', 'BW', '31_3-1',
                                            lognames=mylogs)

    """

    # TODO: replace this with proper class method
    obj = BlockedWell(
        *([0.0] * 3),
        "",
        pd.DataFrame(
            {
                _AttrName.XNAME.value: [],
                _AttrName.YNAME.value: [],
                _AttrName.ZNAME.value: [],
            }
        ),
    )

    _blockedwell_roxapi.import_bwell_roxapi(
        obj,
        project,
        gname,
        bwname,
        wname,
        lognames=lognames,
        ijk=ijk,
        realisation=realisation,
    )

    obj._ensure_consistency()

    return obj


# =============================================================================
# CLASS


class BlockedWell(Well):
    """Class for a blocked well in the XTGeo framework, subclassed from the
    Well class.

    Similar to Wells, the blocked well logs are stored as Pandas dataframe,
    which make manipulation easy and fast.

    For blocked well logs, the numbers of rows cannot be changed if you want to
    save the result in RMS, as this is derived from the grid. Also the blocked well
    icon must exist before save.

    The well trajectory are here represented as logs, and XYZ have magic names as
    default: X_UTME, Y_UTMN, Z_TVDSS, which are the three first Pandas columns.

    Other geometry logs has also 'semi-magic' names:

    M_MDEPTH or Q_MDEPTH: Measured depth, either real/true (M...) or
    quasi computed/estimated (Q...). The Quasi computations may be incorrect for
    all uses, but sufficient for some computations.

    Similar for M_INCL, Q_INCL, M_AZI, Q_AZI.

    I_INDEX, J_INDEX, K_INDEX: They are grid indices. For practical reasons
    they are treated as a CONT logs, since the min/max grid indices usually are
    unknown, and hence making a code index is not trivial.

    All Pandas values (yes, discrete also!) are stored as float32 or float64
    format, and undefined values are Nan. Integers are stored as Float due
    to the lacking support for 'Integer Nan' (currently lacking in Pandas,
    but may come in later Pandas versions).

    Note there is a method that can return a dataframe (copy) with Integer
    and Float columns, see :meth:`get_filled_dataframe`.

    The instance can be made either from file or::

        >>> well1 = xtgeo.blockedwell_from_file(well_dir + '/OP_1.bw')  # RMS ascii well

    If in RMS, instance can be made also from RMS icon::

        well4 = xtgeo.blockedwell_from_roxar(
            project,
            'gridname',
            'bwname',
            'wellname',
        )

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._gridname = None

    @property
    def gridname(self):
        """Returns or set (rename) the grid name that the blocked wells
        belongs to."""
        return self._gridname

    @gridname.setter
    def gridname(self, newname):
        if isinstance(newname, str):
            self._gridname = newname
        else:
            raise ValueError("Input name is not a string.")

    def copy(self):
        newbw = super().copy()

        newbw._gridname = self._gridname

        return newbw

    def to_roxar(self, *args, **kwargs):
        """Set (export) a single blocked well item inside roxar project.

        Note this method works only when inside RMS, or when RMS license is
        activated. RMS will store blocked wells as a Gridmodel feature, not as a
        well.

        Note:
           When project is file path (direct access, outside RMS) then
           ``to_roxar()`` will implicitly do a project save. Otherwise, the project
           will not be saved until the user do an explicit project save action.

        Args:
            project (str or object): Magic object 'project' or file path to project
            gname (str): Name of GridModel icon in RMS
            bwname (str): Name of Blocked Well icon in RMS, usually 'BW'
            wname (str): Name of well, as shown in RMS.
            lognames (list or "all"): List of lognames to include, or use 'all' for
                all current blocked logs for this well (except index logs). Default is
                "all".
            realisation (int): Realisation index (0 is default)
            ijk (bool): If True, then also write special index logs if they exist,
                such as I_INDEX, J_INDEX, K_INDEX, etc. Default is False

        .. versionadded: 2.12

        """
        # TODO: go from *args, **kwargs to keywords
        project = args[0]
        gname = args[1]
        bwname = args[2]
        wname = args[3]
        lognames = kwargs.get("lognames", "all")
        ijk = kwargs.get("ijk", False)
        realisation = kwargs.get("realisation", 0)

        _blockedwell_roxapi.export_bwell_roxapi(
            self,
            project,
            gname,
            bwname,
            wname,
            lognames=lognames,
            ijk=ijk,
            realisation=realisation,
        )
