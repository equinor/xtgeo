"""BlockedWells module, (collection of BlockedWell objects)"""

from xtgeo.common.log import null_logger
from xtgeo.common.xtgeo_dialog import XTGeoDialog

from . import _blockedwells_roxapi
from .blocked_well import blockedwell_from_file
from .wells import Wells

xtg = XTGeoDialog()
logger = null_logger(__name__)


def blockedwells_from_files(
    filelist,
    fformat="rms_ascii",
    mdlogname=None,
    zonelogname=None,
    strict=True,
):
    """Import blocked wells from a list of files (filelist).

    Args:
        filelist (list of str): List with file names
        fformat (str): File format, rms_ascii (rms well) is
            currently supported and default format.
        mdlogname (str): Name of measured depth log, if any
        zonelogname (str): Name of zonation log, if any
        strict (bool): If True, then import will fail if
            zonelogname or mdlogname are asked for but not present
            in wells.

    Example:
        Here the from_file method is used to initiate the object
        directly::

          mywells = BlockedWells(['31_2-6.w', '31_2-7.w', '31_2-8.w'])
    """
    return BlockedWells(
        [
            blockedwell_from_file(
                wfile,
                fformat=fformat,
                mdlogname=mdlogname,
                zonelogname=zonelogname,
                strict=strict,
            )
            for wfile in filelist
        ]
    )


def blockedwells_from_roxar(
    project, gname, bwname, lognames=None, ijk=True
):  # pragma: no cover
    """This makes an instance of a BlockedWells directly from Roxar RMS.

    For arguments, see :meth:`BlockedWells.from_roxar`.

    Note the difference between classes BlockedWell and BlockedWells.

    Example::

        # inside RMS:
        import xtgeo
        mylogs = ['ZONELOG', 'GR', 'Facies']
        mybws = xtgeo.blockedwells_from_roxar(project, 'Simgrid', 'BW',
                                            lognames=mylogs)

    """
    # TODO refactor with class method

    obj = BlockedWells()

    obj._from_roxar(project, gname, bwname, ijk=ijk, lognames=lognames)

    return obj


class BlockedWells(Wells):
    """Class for a collection of BlockedWell objects, for operations that
    involves a number of wells.

    See also the :class:`xtgeo.well.BlockedWell` class.
    """

    def copy(self):
        """Copy a BlockedWells instance to a new unique instance."""

        return BlockedWells([w.copy() for w in self._wells])

    def get_blocked_well(self, name):
        """Get a BlockedWell() instance by name, or None"""
        logger.debug("Calling super...")
        return super().get_well(name)

    def _from_roxar(self, *args, **kwargs):  # pragma: no cover
        """Import (retrieve) blocked wells from roxar project.

        Note this method works only when inside RMS, or when RMS license is
        activated.

        All the wells present in the bwname icon will be imported.

        Args:
            project (str): Magic string 'project' or file path to project
            gname (str): Name of GridModel icon in RMS
            bwname (str): Name of Blocked Well icon in RMS, usually 'BW'
            lognames (list): List of lognames to include, or use 'all' for
                all current blocked logs for this well.
            ijk (bool): If True, then logs with grid IJK as I_INDEX, etc
            realisation (int): Realisation index (0 is default)
        """
        project = args[0]
        gname = args[1]
        bwname = args[2]
        lognames = kwargs.get("lognames")
        ijk = kwargs.get("ijk", True)

        _blockedwells_roxapi.import_bwells_roxapi(
            self, project, gname, bwname, lognames=lognames, ijk=ijk
        )
