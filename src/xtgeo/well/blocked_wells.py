# -*- coding: utf-8 -*-
"""BlockedWells module, (collection of BlockedWell objects)"""

# ======================================================================================


import xtgeo
from .wells import Wells
from .blocked_well import BlockedWell

from . import _blockedwells_roxapi

xtg = xtgeo.common.XTGeoDialog()
logger = xtg.functionlogger(__name__)


def blockedwells_from_roxar(project, gname, bwname, lognames=None, ijk=True):

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

    obj = BlockedWells()

    obj.from_roxar(project, gname, bwname, ijk=ijk, lognames=lognames)

    return obj


class BlockedWells(Wells):
    """Class for a collection of BlockedWell objects, for operations that
    involves a number of wells.

    See also the :class:`xtgeo.well.BlockedWell` class.
    """

    def __init__(self):

        super().__init__()
        self._wells = []  # list of Well objects

    def copy(self):
        """Copy a BlockedWells instance to a new unique instance."""

        new = BlockedWells()

        for well in self._wells:
            newwell = well.copy()
            new._props.append(newwell)

        return new

    def get_blocked_well(self, name):
        """Get a BlockedWell() instance by name, or None"""
        logger.info("Calling super...")
        return super().get_well(name)

    def from_files(
        self,
        filelist,
        fformat="rms_ascii",
        mdlogname=None,
        zonelogname=None,
        strict=True,
        append=True,
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
            append (bool): If True, new wells will be added to existing
                wells.

        Example:
            Here the from_file method is used to initiate the object
            directly::

            >>> mywells = BlockedWells(['31_2-6.w', '31_2-7.w', '31_2-8.w'])
        """

        if not append:
            self._wells = []

        # file checks are done within the Well() class
        for wfile in filelist:
            try:
                wll = BlockedWell(
                    wfile,
                    fformat=fformat,
                    mdlogname=mdlogname,
                    zonelogname=zonelogname,
                    strict=strict,
                )
                self._wells.append(wll)
            except ValueError as err:
                xtg.warn("SKIP this well: {}".format(err))
                continue
        if not self._wells:
            xtg.warn("No wells imported!")

    def from_roxar(self, *args, **kwargs):
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
        lognames = kwargs.get("lognames", None)
        ijk = kwargs.get("ijk", True)

        _blockedwells_roxapi.import_bwells_roxapi(
            self, project, gname, bwname, lognames=lognames, ijk=ijk
        )
