"""BlockedWells module, (collection of BlockedWell objects)"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from xtgeo.common.log import null_logger
from xtgeo.common.xtgeo_dialog import XTGeoDialog
from xtgeo.io._file import FileWrapper

from . import _blockedwells_roxapi
from .blocked_well import blockedwell_from_file
from .wells import Wells

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    import io
    from pathlib import Path

    from xtgeo.common.types import FileLike

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


def blockedwells_from_stacked_file(
    bwfile: FileLike,
    fformat: str | None = None,
    mdlogname: str | None = None,
    zonelogname: str | None = None,
    strict: bool = False,
) -> BlockedWells:
    """Import multiple blocked wells from a single concatenated (stacked) file.

    This function reads files that contain multiple blocked wells in a single file,
    as created by :meth:`BlockedWells.to_stacked_file`.

    For CSV format, expects a WELLNAME column to identify each blocked well.
    For RMS ASCII format, reads multiple blocked well entries, each with its own
    header.

    Args:
        bwfile: File name or stream.
        fformat: File format ('rms_ascii'/'rmswell' or 'csv'). If None, auto-detect
            from file extension or signature.
        mdlogname: Name of measured depth log to use
        zonelogname: Name of zone log to use
        strict: If True, raise error if mdlogname/zonelogname not found

    Returns:
        BlockedWells instance containing all blocked wells from the file.

    Example::

        >>> bwells = xtgeo.blockedwells_from_stacked_file(
        ...     "all_bwells.csv", fformat="csv"
        ... )
        >>> bwells = xtgeo.blockedwells_from_stacked_file("all_bwells.rmswell")

    .. versionadded:: 4.19 (approximate)
    """
    from . import _blockedwells_io_factory

    bwfile_wrapper = FileWrapper(bwfile, mode="rb")
    bwfile_wrapper.check_file(raiseerror=OSError)

    blockedwell_list = _blockedwells_io_factory.blockedwells_from_file(
        bwfile_wrapper, fformat, mdlogname, zonelogname, strict
    )

    return BlockedWells(blockedwell_list)


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

    def to_stacked_file(
        self,
        bwfile: FileLike,
        fformat: str | None = "rms_ascii",
        compression: str | None = "lzf",
    ) -> Path | io.BytesIO | io.StringIO:
        """Export multiple blocked wells to a single concatenated (stacked) file.

        For CSV format, all blocked wells are combined into a single table with a
        WELLNAME column to identify each blocked well.

        For RMS ASCII format, each blocked well is written sequentially in the
        standard RMS well format (with its own header and data section).

        Args:
            bwfile: File name or stream.
            fformat: File format ('rms_ascii'/'rmswell' or 'csv').
                Default is 'rms_ascii'. HDF5 is not supported.
            compression: Not used, kept for API compatibility.

        Returns:
            Path to the file that was written.

        Example::

            >>> bwells = BlockedWells([bwell1, bwell2, bwell3])
            >>> bwells.to_stacked_file("all_bwells.csv", fformat="csv")
            >>> bwells.to_stacked_file("all_bwells.rmswell", fformat="rms_ascii")

        .. versionadded:: 4.19 (approximate)
        """
        from . import _blockedwells_io_factory

        if not self._wells:
            raise ValueError("No blocked wells to export")

        bwfile_wrapper = FileWrapper(bwfile, mode="wb", obj=self)
        bwfile_wrapper.check_folder(raiseerror=OSError)

        _blockedwells_io_factory.blockedwells_to_file(
            self, bwfile_wrapper, fformat, compression
        )

        return bwfile_wrapper.file

    def _from_roxar(
        self,
        project: Any,
        gridname: str,
        bwname: str,
        lognames: str | list[str],
        ijk: bool,
        realisation: int = 0,
    ):
        """Import (retrieve) blocked wells from roxar project.

        Note this method works only when inside RMS, or when RMS license is
        activated.

        All the wells present in the bwname icon will be imported.

        Args:
            project: Magic string 'project' or file path to project
            gridname: Name of GridModel icon in RMS
            bwname: Name of Blocked Well icon in RMS, usually 'BW'
            lognames: List of lognames to include, or use 'all' for
                all current blocked logs for this well.
            ijk: If True, then logs with grid IJK as I_INDEX, etc
            realisation: Realisation index (0 is default)
        """
        _blockedwells_roxapi.import_bwells_roxapi(
            self,
            project,
            gridname,
            bwname,
            lognames=lognames,
            ijk=ijk,
            realisation=realisation,
        )
