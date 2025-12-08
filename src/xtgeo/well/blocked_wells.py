"""BlockedWells module, (collection of BlockedWell objects)"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from xtgeo.io._file import FileFormat, FileWrapper

from . import _blockedwells_roxapi, _well_collection_io
from .blocked_well import blockedwell_from_file
from .wells import Wells

if TYPE_CHECKING:
    from pathlib import Path

    from xtgeo.common.types import FileLike


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
    fformat: str = "rms_ascii_stacked",
    **kwargs,
) -> BlockedWells:
    """Import multiple blocked wells from a single stacked file.

    This function reads a file containing multiple blocked wells in a stacked format,
    where each well's data follows one after another in the same file.

    Args:
        bwfile: Path to file or StringIO object containing multiple blocked wells
        fformat: File format. Supported formats:
            - "rms_ascii_stacked": RMS ASCII well format with multiple wells stacked
            - "csv": CSV file with a well name column identifying each well
        **kwargs: Format-specific parameters.

            CSV format parameters:

            - xname (str): Column name for X coordinates (default: "X_UTME")
            - yname (str): Column name for Y coordinates (default: "Y_UTMN")
            - zname (str): Column name for Z coordinates (default: "Z_TVDSS")
            - i_indexname (str): Column name for I-indices (default: "I_INDEX")
            - j_indexname (str): Column name for J-indices (default: "J_INDEX")
            - k_indexname (str): Column name for K-indices (default: "K_INDEX")
            - wellname_col (str): Column name containing well names
              (default: "WELLNAME")

    Returns:
        BlockedWells instance containing all wells from the file

    Example::

        >>> import xtgeo
        >>> mybws = xtgeo.blockedwells_from_stacked_file("stacked_wells.rmswell")
        >>> print(f"Loaded {len(mybws.wells)} blocked wells")

        For CSV files with custom column names::

        >>> mybws = xtgeo.blockedwells_from_stacked_file(
        ...     "blocked_wells.csv",
        ...     fformat="csv",
        ...     xname="EASTING",
        ...     yname="NORTHING"
        ... )

    .. versionadded:: 4.15
    """
    wfile_obj = FileWrapper(bwfile, mode="r")
    fmt = wfile_obj.fileformat(fformat)

    if fmt == FileFormat.RMSWELL_STACKED:
        well_list = _well_collection_io.import_stacked_rms_ascii(
            bwfile, well_class_name="BlockedWell"
        )
        return BlockedWells(well_list)

    if fmt == FileFormat.CSV:
        well_list = _well_collection_io.import_csv_wells(
            bwfile, well_class_name="BlockedWell", **kwargs
        )
        return BlockedWells(well_list)

    raise ValueError(
        f"Unsupported format: {fformat}. "
        f"Only 'rms_ascii_stacked' and 'csv' are supported."
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
        return super().get_well(name)

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

    def to_stacked_file(
        self,
        wfile: FileLike,  # TODO: check if StringIO is supported!
        fformat: str = "rms_ascii_stacked",
        **kwargs,
    ) -> str | Path:
        """Export multiple blocked wells to a single stacked file.

        This method writes all blocked wells in the collection to a single file where
        each well's data follows one after another.

        Args:
            wfile: Path to the output file
            fformat: File format. Supported formats:
                - "rms_ascii_stacked": RMS ASCII well format with multiple wells stacked
                - "csv": CSV file with a well name column identifying each well
            **kwargs: Format-specific parameters.

                CSV format parameters:

                - xname (str): Column name for X coordinates (default: "X_UTME")
                - yname (str): Column name for Y coordinates (default: "Y_UTMN")
                - zname (str): Column name for Z coordinates (default: "Z_TVDSS")
                - i_indexname (str): Column name for I-indices (default: "I_INDEX")
                - j_indexname (str): Column name for J-indices (default: "J_INDEX")
                - k_indexname (str): Column name for K-indices (default: "K_INDEX")
                - wellname_col (str): Column name containing well names
                  (default: "WELLNAME")
                - Additional keyword arguments are passed to pandas.DataFrame.to_csv()

        Returns:
            Path to the output file

        Raises:
            ValueError: If wells list is empty or unsupported format is specified

        Example::

            >>> import xtgeo
            >>> bwells = xtgeo.blockedwells_from_files([
            ...     'blocked_well1.w', 'blocked_well2.w'
            ... ])
            >>> bwells.to_stacked_file('stacked_blocked_wells.rmswell')

            For CSV format with custom column names::

            >>> bwells.to_stacked_file(
            ...     'blocked_wells.csv',
            ...     fformat='csv',
            ...     xname='EASTING',
            ...     yname='NORTHING',
            ...     i_indexname='ICELL',
            ...     j_indexname='JCELL',
            ...     k_indexname='KCELL'
            ... )

        .. versionadded:: 4.15
        """
        return super().to_stacked_file(wfile, fformat=fformat, **kwargs)

    def to_files(
        self,
        directory: str | Path,
        fformat: str = "rms_ascii",
        template: str = "{wellname}.w",
        **kwargs,
    ) -> list[str]:
        """Export each blocked well to a separate file.

        This method writes each blocked well in the collection to its own individual
        file, as opposed to :meth:`to_stacked_file` which combines all wells into a
        single file.

        Args:
            directory: Directory path where files will be written. Will be created
                if it doesn't exist.
            fformat: File format for export (default: "rms_ascii"). Supported
                formats include "rms_ascii", "csv", "hdf", etc. See
                :meth:`BlockedWell.to_file` for full list of supported formats.
            template: Filename template with {wellname} placeholder
                (default: "{wellname}.w"). The well's name will replace {wellname}
                in the template.
            **kwargs: Additional keyword arguments passed to
                :meth:`BlockedWell.to_file()` for each well. Format-specific options:

                CSV format:

                - xname (str): Column name for X coordinates (default: "X_UTME")
                - yname (str): Column name for Y coordinates (default: "Y_UTMN")
                - zname (str): Column name for Z coordinates (default: "Z_TVDSS")
                - i_indexname (str): Column name for I-indices (default: "I_INDEX")
                - j_indexname (str): Column name for J-indices (default: "J_INDEX")
                - k_indexname (str): Column name for K-indices (default: "K_INDEX")
                - wellname_col (str): Column name for well name
                  (default: "WELLNAME")
                - include_header (bool): Include column headers (default: True)

        Returns:
            List of file paths (as strings) that were created

        Raises:
            ValueError: If wells list is empty

        Example::

            Export all blocked wells to RMS ASCII format::

                >>> import xtgeo
                >>> bwells = xtgeo.blockedwells_from_files(['bw1.w', 'bw2.w', 'bw3.w'])
                >>> files = bwells.to_files('output_dir/', fformat='rms_ascii')
                >>> print(files)
                ['output_dir/bw1.w', 'output_dir/bw2.w', 'output_dir/bw3.w']

            Export with custom filename template::

                >>> files = bwells.to_files(
                ...     'exports/',
                ...     template='{wellname}_blocked.rmswell'
                ... )

            Export to CSV format with custom column names::

                >>> files = bwells.to_files(
                ...     'csv_exports/',
                ...     fformat='csv',
                ...     template='{wellname}.csv',
                ...     xname='EASTING',
                ...     yname='NORTHING',
                ...     i_indexname='ICELL',
                ...     j_indexname='JCELL',
                ...     k_indexname='KCELL',
                ...     include_header=True
                ... )

        .. versionadded:: 4.15
        """
        return super().to_files(directory, fformat=fformat, template=template, **kwargs)
