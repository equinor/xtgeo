"""XTGeo xyz.polygons module, which contains the Polygons class."""

# For polygons, the order of the points sequence is important. In
# addition, a Polygons dataframe _must_ have a INT column called 'POLY_ID'
# which identifies each polygon piece.
from __future__ import annotations

import warnings
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import shapely.geometry as sg

from xtgeo.common._xyz_enum import _AttrName, _XYZType
from xtgeo.common.exceptions import InvalidFileFormatError
from xtgeo.common.log import null_logger
from xtgeo.common.sys import inherit_docstring
from xtgeo.io._file import FileFormat, FileWrapper
from xtgeo.xyz import _xyz_io, _xyz_roxapi

from . import _polygons_oper, _xyz_oper
from ._xyz import XYZ
from ._xyz_io import _convert_idbased_xyz

if TYPE_CHECKING:
    import io
    import pathlib

    from xtgeo.well.well1 import Well

logger = null_logger(__name__)


def _data_reader_factory(file_format: FileFormat):
    if file_format == FileFormat.XYZ:
        return _xyz_io.import_xyz
    if file_format == FileFormat.ZMAP_ASCII:
        return _xyz_io.import_zmap
    if file_format == FileFormat.CSV:
        return _xyz_io.import_csv_polygons
    if file_format == FileFormat.PARQUET:
        return _xyz_io.import_parquet_polygons

    extensions = FileFormat.extensions_string(
        [FileFormat.XYZ, FileFormat.ZMAP_ASCII, FileFormat.CSV, FileFormat.PARQUET]
    )
    raise InvalidFileFormatError(
        f"File format {file_format} is invalid for type Polygons. "
        f"Supported formats are {extensions}."
    )


def _file_importer(
    pfile: str | pathlib.Path | io.BytesIO,
    fformat: str | None = None,
):
    """General function for polygons_from_file"""
    pfile = FileWrapper(pfile)
    fmt = pfile.fileformat(fformat)
    kwargs = _data_reader_factory(fmt)(pfile)

    if "POLY_ID" not in kwargs["values"].columns:
        kwargs["values"]["POLY_ID"] = (
            kwargs["values"].isnull().all(axis=1).cumsum().dropna()
        )
        kwargs["values"].dropna(axis=0, inplace=True)
        kwargs["values"].reset_index(inplace=True, drop=True)
    kwargs["name"] = "poly"
    return kwargs


def _roxar_importer(
    project: str | Any,
    name: str,
    category: str,
    stype: str = "horizons",
    realisation: int = 0,
    attributes: bool | list[str] = False,
):  # pragma: no cover
    kwargs = _xyz_roxapi.load_xyz_from_rms(
        project, name, category, stype, realisation, attributes, _XYZType.POLYGONS.value
    )

    kwargs["name"] = name if name else "poly"
    return kwargs


def _wells_importer(
    wells: list[Well],
    zone: int | None = None,
    resample: int | None = 1,
):
    """Get line segments from a list of wells and a single zone number.

    A future extension is that zone could be a list of zone numbers and/or mechanisms
    to retrieve well segments by other measures, e.g. >= depth.
    """

    dflist = []
    maxid = 0
    for well in wells:
        wp = well.get_zone_interval(zone, resample=resample)
        if wp is not None:
            wp["WellName"] = well.name
            # as well segments may have overlapping POLY_ID:
            wp["POLY_ID"] += maxid
            maxid = wp["POLY_ID"].max() + 1
            dflist.append(wp)

    if not dflist:
        return {}
    dfr = pd.concat(dflist, ignore_index=True)
    dfr.reset_index(inplace=True, drop=True)
    return {
        "values": dfr,
        "attributes": {"WellName": "str"},
    }


def polygons_from_file(pfile: str | pathlib.Path, fformat: str | None = "guess"):
    """Make an instance of a Polygons object directly from file import.

    Supported formats are:

        * 'xyz' or 'pol': Simple XYZ format
        * 'csv': CSV format with mandatory columns X, Y, Z and POLY_ID
        * 'parquet': Parquet format with mandatory columns X, Y, Z and POLY_ID
        * 'zmap': ZMAP line format as exported from RMS (e.g. fault lines)
        * 'guess': Try to choose file format based on extension

    Args:
        pfile (str): Name of file
        fformat (str): See :meth:`Polygons.from_file`

    Example::

        import xtgeo
        mypoly = xtgeo.polygons_from_file('somefile.xyz')
    """
    return Polygons(**_file_importer(pfile, fformat=fformat))


def polygons_from_roxar(
    project: str | Any,
    name: str,
    category: str,
    stype: str | None = "horizons",
    realisation: int | None = 0,
    attributes: bool | list[str] = False,
):  # pragma: no cover
    """Load a Polygons instance from Roxar RMS project.

    Note also that horizon/zone/faults name and category must exists
    in advance, otherwise an Exception will be raised.

    Args:
        project: Name of project (as folder) if outside RMS, or just use the magic
            `project` word if within RMS.
        name: Name of polygons item
        category: For horizons/zones/faults: for example 'DL_depth'
            or use a folder notation on clipboard/general2d_data.
        stype: RMS folder type, 'horizons' (default), 'zones', 'clipboard',
            'faults', 'general2d_data'
        realisation: Realisation number, default is 0
        attributes: Polygons can store an attrubute (e.g. a fault name) per polygon,
            i.e. per "POLY_ID")

    Example::

        import xtgeo
        mysurf = xtgeo.polygons_from_roxar(project, 'TopAare', 'DepthPolys')

    .. versionadded:: 2.19 general2d_data support is added
    .. versionadded:: 3.x support for polygon attributes (other than POLY_ID)
    """

    stype = "horizons" if stype is None else stype
    realisation = realisation if realisation else 0

    return Polygons(
        **_roxar_importer(
            project,
            name,
            category,
            stype,
            realisation,
            attributes,
        )
    )


def polygons_from_wells(
    wells: list[Well],
    zone: int | None = 1,
    resample: int | None = 1,
):
    """Get polygons from wells and a single zone number.

    Args:
        wells: List of XTGeo well objects, a single XTGeo well or a list of well files.
            If a list of well files, the routine will try to load well based on file
            signature and/or extension, but only default settings are applied. Hence
            this is less flexible and more fragile.
        zone: The zone number to extract the linepiece from
        resample: If given, resample every N'th sample to make
            polylines smaller in terms of bits and bytes.
            1 = No resampling, which means just use well sampling (which can be rather
            dense; typically 15 cm).


    Returns:
        None if empty data, otherwise a Polygons() instance.

    Example::

        wells = ["w1.w", "w2.w"]
        points = xtgeo.polygons_from_wells(wells, zone=2)
    """
    return Polygons(**_wells_importer(wells, zone, resample))


def _generate_docstring_polygons(
    xname, yname, zname, pname, hname, dhname, tname, dtname
):
    return f"""
    Class for a Polygons object (connected points) in the XTGeo framework.

    The term Polygons is here used in a wider context, as it includes
    polylines that do not connect into closed polygons. A Polygons
    instance may contain several pieces of polylines/polygons, which are
    identified by POLY_ID.

    The polygons are stored in Python as a Pandas dataframe, which
    allow for flexible manipulation and fast execution.

    A Polygons instance will have 4 mandatory columns; here by default names:

    * {xname} - for X UTM coordinate (Easting)
    * {yname} - For Y UTM coordinate (Northing)
    * {zname} - For depth or property from mean SeaLevel; Depth positive down
    * {pname} - for polygon ID as there may be several polylines segments

    Each Polygons instance can also a name (through the name attribute).
    Default is 'poly'. E.g. if a well fence, it is logical to name the
    instance to be the same as the well name.

    Args:
        values: Provide input values on various forms (list-like or dataframe).
        xname: Name of first (X) mandatory column.
        yname: Name of second (Y) mandatory column.
        zname: Name of third (Z) mandatory column.
        pname: Name of forth (P) mandatory enumerating column.
        hname: Name of cumulative horizontal length, defaults to "{hname}" if
            in dataframe otherwise None.
        dhname: Name of delta horizontal length, defaults to "{dhname}" if in
            dataframe otherwise None.
        tname: Name of cumulative total length, defaults to "{tname}" if in
            dataframe otherwise None.
        dtname: Name of delta total length, defaults to "{dtname}" if in
            dataframe otherwise None.
        attributes: A dictionary for attribute columns as 'name: type', e.g.
            {{"WellName": "str", "IX": "int"}}. This is applied when values are input
            and is to name and type the extra attribute columns in a polygons set.

    Note:
        Most export/import file formats do not support additional attributes; only the
        three first columns (X, Y, Z) are fully supported.
    """


class Polygons(XYZ):
    __doc__ = _generate_docstring_polygons(
        _AttrName.XNAME.value,
        _AttrName.YNAME.value,
        _AttrName.ZNAME.value,
        _AttrName.PNAME.value,
        _AttrName.HNAME.value,
        _AttrName.DHNAME.value,
        _AttrName.TNAME.value,
        _AttrName.DTNAME.value,
    )

    def __init__(
        self,
        values: list | np.ndarray | pd.DataFrame = None,
        xname: str = _AttrName.XNAME.value,
        yname: str = _AttrName.YNAME.value,
        zname: str = _AttrName.ZNAME.value,
        pname: str = _AttrName.PNAME.value,
        hname: str = _AttrName.R_HLEN_NAME.value,
        dhname: str = _AttrName.DHNAME.value,
        tname: str = _AttrName.TNAME.value,
        dtname: str = _AttrName.DTNAME.value,
        name: str = "poly",
        attributes: dict | None = None,
        # from legacy initialization, remove in 4.0, undocumented by purpose:
        fformat: str = "guess",
        filesrc: str = None,
    ):
        self._xyztype: str = _XYZType.POLYGONS.value

        super().__init__(self._xyztype, xname, yname, zname)
        self._pname = pname

        if values is None:
            values = []

        self._attrs = attributes if attributes is not None else {}
        self._filesrc = filesrc

        # additional optional state properties for Polygons
        self._hname = hname
        self._dhname = dhname
        self._tname = tname
        self._dtname = dtname
        self._name = name

        if not isinstance(values, pd.DataFrame):
            self._df = _xyz_io._from_list_like(
                values, self._zname, attributes, self._xyztype
            )
        else:
            self._df = values
            self._dataframe_consistency_check()

    @property
    def name(self):
        """Returns or sets the name of the instance."""
        return self._name

    @name.setter
    def name(self, newname):
        self._name = newname

    @property
    def pname(self):
        return self._pname

    @pname.setter
    def pname(self, name):
        super()._check_name_and_replace(self._pname, name)
        self._pname = name

    @property
    def hname(self):
        """Returns or set the name of the cumulative horizontal length.

        If the column does not exist, None is returned. Default name is H_CUMLEN.

        .. versionadded:: 2.1
        """
        return self._hname

    @hname.setter
    def hname(self, name):
        super()._check_name_and_replace(self._hname, name)
        self._hname = name

    @property
    def dhname(self):
        """Returns or set the name of the delta horizontal length column if it exists.

        If the column does not exist, None is returned. Default name is H_DELTALEN.

        .. versionadded:: 2.1
        """
        return self._dhname

    @dhname.setter
    def dhname(self, name):
        super()._check_name_and_replace(self._dhname, name)
        self._dhname = name

    @property
    def tname(self):
        """Returns or set the name of the cumulative total length column if it exists.

        .. versionadded:: 2.1
        """
        return self._tname

    @tname.setter
    def tname(self, name):
        super()._check_name_and_replace(self._tname, name)
        self._tname = name

    @property
    def dtname(self):
        """Returns or set the name of the delta total length column if it exists.

        .. versionadded:: 2.1
        """
        return self._dtname

    @dtname.setter
    def dtname(self, name):
        super()._check_name_and_replace(self._dtname, name)
        self._dtname = name

    @property
    def dataframe(self) -> pd.DataFrame:
        """Returns or set the Pandas dataframe object."""
        warnings.warn(
            "Direct access to the dataframe property in Polygons class will be "
            "deprecated in xtgeo 5.0. Use `get_dataframe()` instead.",
            PendingDeprecationWarning,
        )
        return self._df

    @dataframe.setter
    def dataframe(self, df):
        warnings.warn(
            "Direct access to the dataframe property in Polygons class will be "
            "deprecated in xtgeo 5.0. Use `set_dataframe()` instead.",
            PendingDeprecationWarning,
        )
        self.set_dataframe(df)

    def get_dataframe(self, copy: bool = True) -> pd.DataFrame:
        """Returns the Pandas dataframe object.

        Args:
            copy: If True, return a deep copy of the dataframe


        .. versionchanged: 3.7  Add keyword `copy` defaulted to True
        """
        return self._df.copy() if copy else self._df

    def set_dataframe(self, df):
        self._df = df.apply(deepcopy)
        self._name_to_none_if_missing()

    def _name_to_none_if_missing(self):
        if self._dtname not in self._df.columns:
            self._dtname = None
        if self._dhname not in self._df.columns:
            self._dhname = None
        if self._tname not in self._df.columns:
            self._tname = None
        if self._hname not in self._df.columns:
            self._hname = None

    # ----------------------------------------------------------------------------------
    # Class methods
    # ----------------------------------------------------------------------------------

    @classmethod
    def boundary_from_points(
        cls,
        points,
        alpha_factor: float | None = 1.0,
        alpha: float | None = None,
        convex: bool = False,
    ):
        """Instantiate polygons from detecting the boundary around points.

        .. image:: images/boundary_polygons.png
           :width: 600
           :align: center

        |

        Args:
            points: The XTGeo Points instance to estimate boundary/boundaries around.
            alpha_factor: The alpha factor is a multiplier to alpha. Normally it will
                be around 1, but can be increased to get a looser boundary. Dependent
                on the points topology, it can also be decreased to some extent.
            alpha: The alpha factor for determine the 'precision' in how to delineate
                the polygon. A large value will produce a smoother polygon. The default
                is to detect the value from the data, but note that this default may be
                far from optimal for you needs. Usually use the ``alpha_factor`` to tune
                the best value. The actual alpha applied in the concave hull algorithm
                is alpha_factor multiplied with alpha.
            convex: If True, then compute a maximum boundary (convex), and note that
                alpha_factor and alpha are not applied in ths case. Default is False.

        Returns:
            A Polygons instance.

        .. versionadded: 3.1.0
        """

        return cls(
            _polygons_oper.boundary_from_points(points, alpha_factor, alpha, convex)
        )

    # ----------------------------------------------------------------------------------
    # Instance methods
    # ----------------------------------------------------------------------------------
    @inherit_docstring(inherit_from=XYZ.protected_columns)
    def protected_columns(self):
        return super().protected_columns() + [self.pname]

    def to_file(
        self,
        pfile,
        fformat: str = "xyz",
        attributes: bool | list[str] = False,
    ):
        """Export Polygons to file.

        Args:
            pfile (str): Name of file
            fformat (str): File format xyz/pol/csv/parquet
            attributes: If True or a list, attributes (additional columns) will be
                preserved if supported by the file format (currently only supported
                by CSV and PARQUET format). The default is False.

        Returns:
            Number of polygon points exported
        """

        return _xyz_io.to_file(self, pfile, fformat=fformat, attributes=attributes)

    def to_roxar(
        self,
        project,
        name,
        category,
        stype="horizons",
        realisation=0,
        attributes=False,
    ):  # pragma: no cover
        """Export (store) a Polygons item to a Roxar RMS project.

        The export to the RMS project can be done either within the project
        or outside the project.

        Note also that horizon/zone name and category must exists in advance,
        otherwise an Exception will be raised.

        Note:
            When project is file path (direct access, outside RMS) then
            ``to_roxar()`` will implicitly do a project save. Otherwise, the project
            will not be saved until the user do an explicit project save action.

        Args:
            project (str or special): Name of project (as folder) if
                outside RMS, og just use the magic project word if within RMS.
            name (str): Name of polygons item
            category (str): For horizons/zones/faults: for example 'DL_depth' and use
                a folder notation for clipboard/general2d_data
            stype (str): RMS folder type, 'horizons' (default), 'zones'
                or 'faults' or 'clipboard'  (in prep: well picks)
            realisation (int): Realisation number, default is 0
            attributes (bool): If True, attributes will be preserved (from RMS 13)

        Raises:
            ValueError: Various types of invalid inputs.
            NotImplementedError: Not supported in this ROXAPI version

        Note:
            Setting (storing) polygons with attributes is not supported in RMSAPI.

        .. versionadded:: 2.19 general2d_data support is added
        .. versionadded:: 4.X Added attributes
        """

        _xyz_roxapi.save_xyz_to_rms(
            self,
            project,
            name,
            category,
            stype,
            None,
            realisation,
            attributes,
        )

    def copy(self):
        """Returns a deep copy of an instance"""
        mycopy = self.__class__()
        mycopy._xyztype = self._xyztype
        mycopy._df = self._df.apply(deepcopy)  # df.copy() is not fully deep!
        mycopy._xname = self._xname
        mycopy._yname = self._yname
        mycopy._zname = self._zname
        mycopy._pname = self._pname
        mycopy._hname = self._hname
        mycopy._dhname = self._dhname
        mycopy._tname = self._tname
        mycopy._dtname = self._dtname

        if self._attrs:
            mycopy._attrs = dict(self._attrs.items())

        return mycopy

    def get_xyz_dataframe(self):
        """Get a dataframe copy from the Polygons points with no ID column.

        Convert from POLY_ID based to XYZ, where a new polygon is marked with a 999
        value as flag.
        """
        return _convert_idbased_xyz(self, self.get_dataframe())

    def get_shapely_objects(self):
        """Returns a list of Shapely LineString objects, one per POLY_ID.

        .. versionadded:: 2.1
        """
        spolys = []
        idgroups = self.get_dataframe(copy=False).groupby(self.pname)

        for _idx, grp in idgroups:
            pxcor = grp[self.xname].values
            pycor = grp[self.yname].values
            pzcor = grp[self.zname].values
            spoly = sg.LineString(np.stack([pxcor, pycor, pzcor], axis=1))
            spolys.append(spoly)

        return spolys

    @inherit_docstring(inherit_from=XYZ.get_boundary)
    def get_boundary(self):
        return super().get_boundary()

    @inherit_docstring(inherit_from=XYZ.get_xyz_arrays)
    def get_xyz_arrays(self):
        return super().get_xyz_arrays()

    def simplify(
        self, tolerance: float | None = 0.1, preserve_topology: bool | None = True
    ) -> bool:
        """Simply a polygon, i.e. remove unneccesary points.

        This is based on `Shapely's simplify() method
        <https://shapely.readthedocs.io/en/latest/manual.html#object.simplify>`_

        Args:
            tolerance: Cf. Shapely's documentation
            preserve_topology: Default is True, if False a faster algorithm is applied

        Returns:
            True if simplification is achieved. The polygons instance is
            updated in-place.

        .. versionadded: 3.1


        """

        return _polygons_oper.simplify_polygons(self, tolerance, preserve_topology)

    def filter_byid(self, polyid=None):
        """Remove all line segments not in polyid.

        The instance is updated in-place.

        Args:
            polyid (int or list of int): Which ID(s) to keep, None means use first.

        Example::

            mypoly.filter_byid(polyid=[2, 4])  # keep POLY_ID 2 and 4

        .. versionadded:: 2.1
        """
        dataframe = self.get_dataframe()
        if polyid is None:
            polyid = int(dataframe[self.pname].iloc[0])

        if not isinstance(polyid, list):
            polyid = [polyid]

        dflist = []
        for pid in polyid:
            dflist.append(dataframe[dataframe[self.pname] == pid])

        dataframe = pd.concat(dflist)
        self.set_dataframe(dataframe)

    def tlen(self, tname="T_CUMLEN", dtname="T_DELTALEN", atindex=0):
        """Compute and add or replace columns for cum. total 3D length and delta length.

        The instance is updated in-place.

        Args:
            tname (str): Name of cumulative total length. Default is T_CUMLEN.
            dtname (str): Name of delta length column. Default is T_DELTALEN.
            atindex (int): Which index which shall be 0.0 for cumulative length.

        .. versionadded:: 2.1
        """
        _xyz_oper.tlen(self, tname=tname, dtname=dtname, atindex=atindex)

    def hlen(self, hname="H_CUMLEN", dhname="H_DELTALEN", atindex=0):
        """Compute and add/replace columns for cum. horizontal length and delta length.

        The instance is updated in-place.

        Args:
            hname (str): Name of cumulative horizontal length. Default is H_CUMLEN.
            dhname (str): Name of delta length column. Default is H_DELTALEN.
            atindex (int): Which index which shall be 0.0 for cumulative length.

        .. versionadded:: 2.1
        """
        _xyz_oper.hlen(self, hname=hname, dhname=dhname, atindex=atindex)

    def extend(self, distance, nsamples=1, mode2d=True):
        """Extend polyline by `distance` at both ends, nsmaples times.

        The instance is updated in-place.

        Args:
            distance (float): The horizontal distance (sampling) to extend
            nsamples (int): Number of samples to extend.
            mode2d (bool): XY extension (only True is supported)

        .. versionadded:: 2.1
        """
        _xyz_oper.extend(self, distance, nsamples, mode2d)

    def rescale(self, distance, addlen=False, kind="simple", mode2d=True):
        """Rescale (resample) by using a new increment.

        The increment (distance) may be a horizontal or a True 3D
        distance dependent on mode2d.

        The instance is updated in-place.

        If the distance is larger than the total input poly-line length,
        nothing is done. Note that the result distance may differ from the
        requested distance caused to rounding to fit original length.

        Hence actual distance is input distance +- 50%.

        Args:
            distance (float): New distance between points
            addlen (str): If True, total and horizontal cum. and delta length
                columns will be added.
            kind (str): What kind of rescaling: slinear/cubic/simple
            mode2d (bool): The distance may be a 2D (XY) ora 3D (XYZ) mode.

        .. versionchanged:: 2.1 a new algorithm

        """
        _xyz_oper.rescale_polygons(
            self, distance=distance, addlen=addlen, kind=kind, mode2d=mode2d
        )

    def get_fence(
        self, distance=20, atleast=5, nextend=2, name=None, asnumpy=True, polyid=None
    ):
        """Extracts a fence with constant horizontal sampling.

        Additonal H_CUMLEN and H_DELTALEN vectors will be added, suitable for
        X sections.

        Args:
            distance (float): New horizontal distance between points
            atleast (int): Minimum number of points. If the true length/atleast is
                less than distance, than distance will be be reset to
                length/atleast. Values below 3 are not permitted
            nextend (int): Number of samples to extend at each end. Note that
                in case of internal resetting of distance (due to 'atleast'), then
                nextend internally will be modified in order to fulfill the
                initial intention. Hence keep distance*nextend as target.
            name (str): Name of polygon (if asnumpy=False)
            asnumpy (bool): Return a [:, 5] numpy array with
                columns X.., Y.., Z.., HLEN, dH
            polyid (int): Which POLY_ID to use. Default (if None) is to use the
                first found.

        Returns:
            A numpy array (if asnumpy=True) or a new Polygons() object

        .. versionadded:: 2.1
        """
        logger.info("Getting fence within a Polygons instance...")
        return _xyz_oper.get_fence(
            self,
            distance=distance,
            atleast=atleast,
            nextend=nextend,
            name=name,
            asnumpy=asnumpy,
            polyid=polyid,
        )

    # ==================================================================================
    # Plotting
    # ==================================================================================

    def quickplot(
        self,
        filename=None,
        others=None,
        title="QuickPlot for Polygons",
        subtitle=None,
        infotext=None,
        linewidth=1.0,
        color="r",
    ):
        """Simple plotting of polygons using matplotlib.

        Args:
            filename (str): Name of plot file; None will plot to screen.
            others (list of Polygons): List of other polygon instances to plot
            title (str): Title of plot
            subtitle (str): Subtitle of plot
            infotext (str): Additonal info on plot.
            linewidth (float): Width of line.
            color (str): Name of color (may use matplotib shortcuts, e.g. 'r' for 'red')
        """
        import xtgeoviz.plot

        mymap = xtgeoviz.plot.Map()
        mymap.canvas(title=title, subtitle=subtitle, infotext=infotext)

        if others:
            for other in others:
                lwid = linewidth / 2.0
                mymap.plot_polygons(
                    other, idname=other.pname, linewidth=lwid, color="black"
                )

        mymap.plot_polygons(self, idname=self.pname, linewidth=linewidth, color=color)

        if filename is None:
            mymap.show()
        else:
            mymap.savefig(filename)
