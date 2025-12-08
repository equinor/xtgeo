"""Wells module, which has the Wells class (collection of Well objects)"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from xtgeo.common.log import null_logger
from xtgeo.common.xtgeo_dialog import XTGDescription, XTGeoDialog
from xtgeo.io._file import FileFormat, FileWrapper

from . import _well_collection_io, _wells_utils
from .well1 import Well, well_from_file

if TYPE_CHECKING:
    from xtgeo.common.types import FileLike

xtg = XTGeoDialog()
logger = null_logger(__name__)


def wells_from_files(filelist, *args, **kwargs):
    """Import wells from a list of files (filelist).

    Creates a Wells object from a list of filenames. Remaining arguments are
        the same as :func:`xtgeo.well_from_file`.

    Args:
        filelist (list of filenames): List with file names

    Example:
        Here the from_file method is used to initiate the object
        directly::

            >>> mywells = Wells(
            ...     [well_dir + '/OP_1.w', well_dir + '/OP_2.w']
            ... )
    """
    return Wells([well_from_file(wfile, *args, **kwargs) for wfile in filelist])


def wells_from_stacked_file(
    wfile: FileLike,
    fformat: str = "rms_ascii_stacked",
    **kwargs,
) -> Wells:
    """Import multiple wells from a single stacked file.

    This function reads a file containing multiple wells in a stacked format,
    where each well's data follows one after another in the same file.

    Args:
        wfile: Path to file or StringIO object containing multiple wells
        fformat: File format. Supported formats:
            - "rms_ascii_stacked": RMS ASCII well format with multiple wells stacked
            - "csv": CSV file with a well name column identifying each well
        **kwargs: Format-specific parameters.

            CSV format parameters:

            - xname (str): Column name for X coordinates (default: "X_UTME")
            - yname (str): Column name for Y coordinates (default: "Y_UTMN")
            - zname (str): Column name for Z coordinates (default: "Z_TVDSS")
            - wellname_col (str): Column name containing well names
              (default: "WELLNAME")

    Returns:
        Wells instance containing all wells from the file

    Example::

        >>> import xtgeo
        >>> mywells = xtgeo.wells_from_stacked_file("stacked_wells.rmswell")
        >>> print(f"Loaded {len(mywells.wells)} wells")

        For CSV files with custom column names::

        >>> mywells = xtgeo.wells_from_stacked_file(
        ...     "wells.csv",
        ...     fformat="csv",
        ...     xname="EASTING",
        ...     yname="NORTHING"
        ... )

    .. versionadded:: 4.15
    """
    wfile_obj = FileWrapper(wfile, mode="r")
    fmt = wfile_obj.fileformat(fformat)

    if fmt == FileFormat.RMSWELL_STACKED:
        well_list = _well_collection_io.import_stacked_rms_ascii(
            wfile, well_class_name="Well"
        )
        return Wells(well_list)

    if fmt == FileFormat.CSV:
        well_list = _well_collection_io.import_csv_wells(
            wfile, well_class_name="Well", **kwargs
        )
        return Wells(well_list)

    raise ValueError(
        f"Unsupported format: {fformat}. "
        f"Only 'rms_ascii_stacked' and 'csv' are supported."
    )


class Wells:
    """Class for a collection of Well objects, for operations that involves
    a number of wells.

    See also the :class:`xtgeo.well.Well` class.

    Args:
        wells: The list of Well objects.
    """

    def __init__(self, wells: list[Well] = None):
        if wells is None:
            self._wells = []
        else:
            self._wells = wells

    @property
    def names(self):
        """Returns a list of well names (read only).

        Example::

            namelist = wells.names
            for prop in namelist:
                print ('Well name is {}'.format(name))

        """
        return [w.name for w in self._wells]

    @property
    def wells(self):
        """Returns or sets a list of XTGeo Well objects, None if empty."""
        if not self._wells:
            return None

        return self._wells

    @wells.setter
    def wells(self, well_list):
        for well in well_list:
            if not isinstance(well, Well):
                raise ValueError("Well in list not valid Well object")

        self._wells = well_list

    def describe(self, flush=True):
        """Describe an instance by printing to stdout"""

        dsc = XTGDescription()
        dsc.title(f"Description of {self.__class__.__name__} instance")
        dsc.txt("Object ID", id(self))

        dsc.txt("Wells", self.names)

        if flush:
            dsc.flush()
            return None

        return dsc.astext()

    def __iter__(self):
        return iter(self.wells)

    def copy(self):
        """Copy a Wells instance to a new unique instance (a deep copy)."""

        return Wells(self._wells.copy())

    def get_well(self, name):
        """Get a Well() instance by name, or None"""

        logger.debug("Asking for a well with name %s", name)
        for well in self._wells:
            if well.name == name:
                return well
        return None

    # not having this as property but a get_ .. is intended, for flexibility
    def get_dataframe(self, filled=False, fill_value1=-999, fill_value2=-9999):
        """Get a big dataframe for all wells or blocked wells in instance,
        with well name as first column

        Args:
            filled (bool): If True, then NaN's are replaces with values
            fill_value1 (int): Only applied if filled=True, for logs that
                have missing values
            fill_value2 (int): Only applied if filled=True, when logs
                are missing completely for that well.
        """
        logger.debug("Ask for big dataframe for all wells")

        bigdflist = []
        for well in self._wells:
            dfr = well.get_dataframe()
            dfr["WELLNAME"] = well.name
            logger.debug(well.name)
            if filled:
                dfr = dfr.fillna(fill_value1)
            bigdflist.append(dfr)

        dfr = pd.concat(bigdflist, ignore_index=True, sort=True)
        # the concat itself may lead to NaN's:
        if filled:
            dfr = dfr.fillna(fill_value2)

        spec_order = [
            "WELLNAME",
            self._wells[0].xname,  # use the names in the first well as column names
            self._wells[0].yname,
            self._wells[0].zname,
        ]
        return dfr[spec_order + [col for col in dfr if col not in spec_order]]

    def quickplot(self, filename=None, title="QuickPlot"):
        """Fast plot of wells using matplotlib.

        Args:
            filename (str): Name of plot file; None will plot to screen.
            title (str): Title of plot

        """
        import xtgeoviz.plot

        mymap = xtgeoviz.plot.Map()

        mymap.canvas(title=title)

        mymap.plot_wells(self)

        if filename is None:
            mymap.show()
        else:
            mymap.savefig(filename)

    def limit_tvd(self, tvdmin, tvdmax):
        """Limit TVD to be in range tvdmin, tvdmax for all wells"""
        for well in self.wells:
            well.limit_tvd(tvdmin, tvdmax)

    def downsample(self, interval=4, keeplast=True):
        """Downsample by sampling every N'th element (coarsen only), all
        wells.
        """

        for well in self.wells:
            well.downsample(interval=interval, keeplast=keeplast)

    def wellintersections(self, wfilter=None, showprogress=False):
        """Get intersections between wells, return as dataframe table.

        Notes on wfilter: A wfilter is settings to improve result. In
        particular to remove parts of trajectories that are parallel.

        wfilter = {'parallel': {'xtol': 4.0, 'ytol': 4.0, 'ztol':2.0,
                                'itol':10, 'atol':2}}

        Here xtol is tolerance in X coordinate; further Y tolerance,
        Z tolerance, (I)nclination tolerance, and (A)zimuth tolerance.

        Args:
            tvdrange (tuple of floats): Search interval. One is often just
                interested in the reservoir section.
            wfilter (dict): A dictionrary for filter options, in order to
                improve result. See example above.
            showprogress (bool): Will show progress to screen if enabled.

        Returns:
            A Pandas dataframe object, with columns WELL, CWELL and UTMX UTMY
                TVD coordinates for CWELL where CWELL crosses WELL,
                and also MDEPTH for the WELL.
        """

        return _wells_utils.wellintersections(
            self, wfilter=wfilter, showprogress=showprogress
        )

    def to_stacked_file(
        self,
        wfile: FileLike,
        fformat: str = "rms_ascii_stacked",
        **kwargs,
    ) -> FileLike:
        """Export multiple wells to a single stacked file.

        This method writes all wells in the collection to a single file where
        each well's data follows one after another.

        Args:
            wfile: Path to the output file or StringIO object
            fformat: File format. Supported formats:
                - "rms_ascii_stacked": RMS ASCII well format with multiple wells stacked
                - "csv": CSV file with a well name column identifying each well
            **kwargs: Format-specific parameters.

                CSV format parameters:

                - xname (str): Column name for X coordinates (default: "X_UTME")
                - yname (str): Column name for Y coordinates (default: "Y_UTMN")
                - zname (str): Column name for Z coordinates (default: "Z_TVDSS")
                - wellname_col (str): Column name containing well names
                  (default: "WELLNAME")
                - Additional keyword arguments are passed to pandas.DataFrame.to_csv()

        Returns:
            Path to the output file

        Raises:
            ValueError: If wells list is empty or unsupported format is specified

        Example::

            >>> import xtgeo
            >>> wells = xtgeo.wells_from_files([
            ...     'well1.w', 'well2.w', 'well3.w'
            ... ])
            >>> wells.to_stacked_file('stacked_wells.rmswell')

            For CSV format with custom column names::

            >>> wells.to_stacked_file(
            ...     'wells.csv',
            ...     fformat='csv',
            ...     xname='EASTING',
            ...     yname='NORTHING',
            ...     wellname_col='WELL_NAME'
            ... )

        .. versionadded:: 4.15
        """
        if not self._wells:
            raise ValueError("Cannot export empty wells list")

        wfile_obj = FileWrapper(wfile, mode="w")
        fmt = wfile_obj.fileformat(fformat)

        if fmt == FileFormat.RMSWELL_STACKED:
            _well_collection_io.export_stacked_rms_ascii(self._wells, wfile_obj.file)
        elif fmt == FileFormat.CSV:
            _well_collection_io.export_csv_wells(self._wells, wfile_obj.file, **kwargs)
        else:
            raise ValueError(
                f"Unsupported format: {fformat}. "
                f"Only 'rms_ascii_stacked' and 'csv' are supported."
            )

        return wfile_obj.file

    def to_files(
        self,
        directory: str | Path,
        fformat: str = "rms_ascii",
        template: str = "{wellname}.w",
        **kwargs,
    ) -> list[str]:
        """Export each well to a separate file.

        This method writes each well in the collection to its own individual file,
        as opposed to :meth:`to_stacked_file` which combines all wells into a
        single file.

        Args:
            directory: Directory path where files will be written. Will be created
                if it doesn't exist.
            fformat: File format for export (default: "rms_ascii"). Supported
                formats include "rms_ascii", "csv", "hdf", etc. See
                :meth:`Well.to_file` for full list of supported formats.
            template: Filename template with {wellname} placeholder
                (default: "{wellname}.w"). The well's name will replace {wellname}
                in the template.
            **kwargs: Additional keyword arguments passed to :meth:`Well.to_file()`
                for each well. Format-specific options:

                CSV format:

                - xname (str): Column name for X coordinates (default: "X_UTME")
                - yname (str): Column name for Y coordinates (default: "Y_UTMN")
                - zname (str): Column name for Z coordinates (default: "Z_TVDSS")
                - wellname_col (str): Column name for well name
                  (default: "WELLNAME")
                - include_header (bool): Include column headers (default: True)

        Returns:
            List of file paths (as strings) that were created

        Raises:
            ValueError: If wells list is empty

        Example::

            Export all wells to RMS ASCII format::

                >>> import xtgeo
                >>> wells = xtgeo.wells_from_files(['well1.w', 'well2.w', 'well3.w'])
                >>> files = wells.to_files('output_dir/', fformat='rms_ascii')
                >>> print(files)
                ['output_dir/well1.w', 'output_dir/well2.w', 'output_dir/well3.w']

            Export with custom filename template::

                >>> files = wells.to_files(
                ...     'exports/',
                ...     template='{wellname}_export.rmswell'
                ... )

            Export to CSV format with custom column names::

                >>> files = wells.to_files(
                ...     'csv_exports/',
                ...     fformat='csv',
                ...     template='{wellname}.csv',
                ...     xname='EASTING',
                ...     yname='NORTHING',
                ...     include_header=True
                ... )

        .. versionadded:: 4.15
        """
        if not self._wells:
            raise ValueError("Cannot export empty wells list")

        output_dir = Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        created_files = []
        for well in self._wells:
            # Generate filename from template
            filename = template.format(wellname=well.name)
            filepath = output_dir / filename

            # Export the well
            well.to_file(filepath, fformat=fformat, **kwargs)
            created_files.append(str(filepath))

            logger.debug(f"Exported well '{well.name}' to {filepath}")

        logger.debug(f"Exported {len(created_files)} wells to {output_dir}")
        return created_files
