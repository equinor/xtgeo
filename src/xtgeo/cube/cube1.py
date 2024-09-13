"""Module for a seismic (or whatever) cube."""

from __future__ import annotations

import numbers
import warnings
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from xtgeo.common.constants import UNDEF
from xtgeo.common.exceptions import InvalidFileFormatError
from xtgeo.common.log import null_logger
from xtgeo.common.sys import generic_hash
from xtgeo.common.types import Dimensions
from xtgeo.common.xtgeo_dialog import XTGDescription
from xtgeo.grid3d.grid import grid_from_cube
from xtgeo.io._file import FileFormat, FileWrapper
from xtgeo.metadata.metadata import MetaDataRegularCube
from xtgeo.xyz.polygons import Polygons

from . import (
    _cube_export,
    _cube_import,
    _cube_roxapi,
    _cube_utils,
    _cube_window_attributes,
)

if TYPE_CHECKING:
    from xtgeo.surface.regular_surface import RegularSurface

logger = null_logger(__name__)


def _data_reader_factory(fmt: FileFormat):
    if fmt == FileFormat.SEGY:
        return _cube_import.import_segy
    if fmt == FileFormat.STORM:
        return _cube_import.import_stormcube
    if fmt == FileFormat.XTG:
        return _cube_import.import_xtgregcube

    extensions = FileFormat.extensions_string(
        [FileFormat.SEGY, FileFormat.STORM, FileFormat.XTG]
    )
    raise InvalidFileFormatError(
        f"File format {fmt} is invalid for type Cube. "
        f"Supported formats are {extensions}."
    )


def cube_from_file(mfile, fformat="guess"):
    """This makes an instance of a Cube directly from file import.

    Args:
        mfile (str): Name of file
        fformat (str): See :meth:`Cube.from_file`

    Example::

        >>> import xtgeo
        >>> mycube = xtgeo.cube_from_file(cube_dir + "/ib_test_cube2.segy")
    """
    return Cube._read_file(mfile, fformat)


def cube_from_roxar(project, name, folder=None):
    """This makes an instance of a Cube directly from roxar input.

    The folder is a string on form "a" or "a/b" if subfolders are present

    Example::

        import xtgeo
        mycube = xtgeo.cube_from_roxar(project, "DepthCube")

    """
    # this is certainly hackish, and shall be rewritten to a proper class method
    obj = Cube(ncol=9, nrow=9, nlay=9, xinc=9.99, yinc=9.99, zinc=9.99)  # dummy
    _cube_roxapi.import_cube_roxapi(obj, project, name, folder=folder)
    obj._metadata.required = obj
    return obj


class Cube:
    """Class for a (seismic) cube in the XTGeo framework.

    The values are stored as a 3D numpy array (4 bytes; float32 is default),
    with internal C ordering (nlay fastest).

    See :func:`xtgeo.cube_from_file` for importing cubes from e.g. segy files.

    See also Cube section in documentation: docs/datamodel.rst

    Examples::

        >>> import xtgeo
        >>> # a user defined cube:
        >>> mycube = xtgeo.Cube(
        ...     xori=100.0,
        ...     yori=200.0,
        ...     zori=150.0,
        ...     ncol=40,
        ...     nrow=30,
        ...     nlay=10,
        ...     rotation=30,
        ...     values=0
        ... )

    Args:
      xori: Origin in Easting coordinate
      yori: Origin in Northing coordinate
      zori: Origin in Depth coordinate, where depth is positive down
      ncol: Number of columns
      nrow: Number of columns
      nlay: Number of layers, starting from top
      rotation: Cube rotation, X axis is applied and "school-wise" rotation,
                     anti-clock in degrees
      values: Numpy array with shape (ncol, nrow, nlay), C order, np.float32
      ilines: 1D numpy array with ncol elements, aka INLINES array, defaults to arange
      xlines: 1D numpy array with nrow elements, aka XLINES array, defaults to arange
      segyfile: Name of source segyfile if any
      filesrc: String: Source file if any
      yflip: Normally 1; if -1 Y axis is flipped --> from left-handed (1) to
                     right handed (-1). Right handed cubes are common.

    """

    def __init__(
        self,
        ncol,
        nrow,
        nlay,
        xinc,
        yinc,
        zinc,
        xori=0.0,
        yori=0.0,
        zori=0.0,
        yflip=1,
        values=0.0,
        rotation=0.0,
        zflip=1,
        ilines=None,
        xlines=None,
        traceidcodes=None,
        segyfile=None,
        filesrc=None,
    ):
        """Initiate a Cube instance."""

        self._filesrc = filesrc
        self._xori = xori
        self._yori = yori
        self._zori = zori
        self._ncol = ncol
        self._nrow = nrow
        self._nlay = nlay
        self._xinc = xinc
        self._yinc = yinc
        self._zinc = zinc
        self._yflip = yflip
        self._zflip = zflip  # currently not in use
        self._rotation = rotation

        # input values can be "list-like" or scalar
        self._values = self._ensure_correct_values(values)

        if ilines is None:
            self._ilines = ilines or np.array(range(1, self._ncol + 1), dtype=np.int32)
        else:
            self._ilines = ilines
        if xlines is None:
            self._xlines = np.array(range(1, self._nrow + 1), dtype=np.int32)
        else:
            self._xlines = xlines
        if traceidcodes is None:
            self._traceidcodes = np.ones((self._ncol, self._nrow), dtype=np.int32)
        else:
            self._traceidcodes = traceidcodes
        self._segyfile = segyfile
        self.undef = UNDEF

        self._metadata = MetaDataRegularCube()
        self._metadata.required = self

    def __repr__(self):
        """The __repr__ method."""
        avg = self.values.mean()
        return (
            f"{self.__class__} (ncol={self.ncol!r}, nrow={self.nrow!r}, "
            f"nlay={self.nlay!r}, original file: {self._filesrc}), "
            f"average {avg}, ID=<{id(self)}>"
        )

    def __str__(self):
        """The __str__ method for pretty print."""
        return self.describe(flush=False)

    @property
    def metadata(self):
        """Return metadata object instance of type MetaDataRegularSurface."""
        return self._metadata

    @metadata.setter
    def metadata(self, obj):
        # The current metadata object can be replaced. This is a bit dangerous so
        # further check must be done to validate. TODO.
        if not isinstance(obj, MetaDataRegularCube):
            raise ValueError("Input obj not an instance of MetaDataRegularCube")

        self._metadata = obj  # checking is currently missing! TODO

    @property
    def ncol(self):
        """The NCOL (NX or I dir) number (read-only)."""
        return self._ncol

    @property
    def nrow(self):
        """The NROW (NY or J dir) number (read-only)."""
        return self._nrow

    @property
    def nlay(self):
        """The NLAY (or NZ or K dir) number (read-only)."""
        return self._nlay

    @property
    def dimensions(self):
        """NamedTuple: The cube dimensions with 3 integers (read only)."""
        return Dimensions(self._ncol, self._nrow, self._nlay)

    @property
    def xori(self):
        """The XORI (origin corner) coordinate."""
        return self._xori

    @xori.setter
    def xori(self, val):
        logger.warning("Changing xori is risky!")
        self._xori = val

    @property
    def yori(self):
        """The YORI (origin corner) coordinate."""
        return self._yori

    @yori.setter
    def yori(self, val):
        logger.warning("Changing yori is risky!")
        self._yori = val

    @property
    def zori(self):
        """The ZORI (origin corner) coordinate."""
        return self._zori

    @zori.setter
    def zori(self, val):
        logger.warning("Changing zori is risky!")
        self._zori = val

    @property
    def xinc(self):
        """The XINC (increment X) as property."""
        return self._xinc

    @xinc.setter
    def xinc(self, val):
        logger.warning("Changing xinc is risky!")
        self._xinc = val

    @property
    def yinc(self):
        """The YINC (increment Y)."""
        return self._yinc

    @yinc.setter
    def yinc(self, val):
        logger.warning("Changing yinc is risky!")
        self._yinc = val

    @property
    def zinc(self):
        """The ZINC (increment Z)."""
        return self._zinc

    @zinc.setter
    def zinc(self, val):
        logger.warning("Changing zinc is risky!")
        self._zinc = val

    @property
    def rotation(self):
        """The rotation, anticlock from X axis in degrees."""
        return self._rotation

    @rotation.setter
    def rotation(self, val):
        logger.warning("Changing rotation is risky!")
        self._rotation = val

    @property
    def ilines(self):
        """The inlines numbering vector."""
        return self._ilines

    @ilines.setter
    def ilines(self, values):
        self._ilines = values

    @property
    def xlines(self):
        """The xlines numbering vector."""
        return self._xlines

    @xlines.setter
    def xlines(self, values):
        self._xlines = values

    @property
    def zslices(self):
        """Return the time/depth slices as an int array (read only)."""
        return np.array(range(self.nlay))  # This is a derived property

    @property
    def traceidcodes(self):
        """The trace identifaction codes array (ncol, nrow)."""
        return self._traceidcodes

    @traceidcodes.setter
    def traceidcodes(self, values):
        if isinstance(values, (int, str)):
            self._traceidcodes = np.full((self.ncol, self.nrow), values, dtype=np.int32)
        else:
            if isinstance(values, list):
                values = np.array(values, np.int32)
            self._traceidcodes = values.reshape(self.ncol, self.nrow)

    @property
    def yflip(self):
        """The YFLIP indicator, 1 is normal, -1 means Y flipped.

        YFLIP = 1 means a LEFT HANDED coordinate system with Z axis
        positive down, while inline (col) follow East (X) and xline (rows)
        follows North (Y), when rotation is zero.
        """
        return self._yflip

    @property
    def zflip(self):
        """The ZFLIP indicator, 1 is normal, -1 means Z flipped.

        ZFLIP = 1 and YFLIP = 1 means a LEFT HANDED coordinate system with Z axis
        positive down, while inline (col) follow East (X) and xline (rows)
        follows North (Y), when rotation is zero.
        """
        return self._zflip

    @property
    def segyfile(self):
        """The input segy file name (str), if any (or None) (read-only)."""
        return self._segyfile

    @property
    def filesrc(self):
        """The input file name (str), if any (or None) (read-only)."""
        return self._filesrc

    @filesrc.setter
    def filesrc(self, name):
        self._filesrc = name

    @property
    def values(self):
        """The values, as a 3D numpy (ncol, nrow, nlay), 4 byte float."""
        return self._values

    @values.setter
    def values(self, values):
        self._values = self._ensure_correct_values(values)

    # =========================================================================
    # Describe
    # =========================================================================

    def generate_hash(self, hashmethod="md5"):
        """Return a unique hash ID for current instance.

        See :meth:`~xtgeo.common.sys.generic_hash()` for documentation.

        .. versionadded:: 2.14
        """
        required = (
            "ncol",
            "nrow",
            "nlay",
            "xori",
            "yori",
            "zori",
            "xinc",
            "yinc",
            "zinc",
            "yflip",
            "zflip",
            "rotation",
            "values",
            "ilines",
            "xlines",
            "traceidcodes",
        )

        gid = ""
        for req in required:
            gid += f"{getattr(self, '_' + req)}"

        return generic_hash(gid, hashmethod=hashmethod)

    def describe(self, flush=True):
        """Describe an instance by printing to stdout or return.

        Args:
            flush (bool): If True, description is printed to stdout.
        """
        dsc = XTGDescription()
        dsc.title("Description of Cube instance")
        dsc.txt("Object ID", id(self))
        dsc.txt("File source", self._filesrc)
        dsc.txt("Shape: NCOL, NROW, NLAY", self.ncol, self.nrow, self.nlay)
        dsc.txt("Origins XORI, YORI, ZORI", self.xori, self.yori, self.zori)
        dsc.txt("Increments XINC YINC ZINC", self.xinc, self.yinc, self.zinc)
        dsc.txt("Rotation (anti-clock from X)", self.rotation)
        dsc.txt("YFLIP flag", self.yflip)
        np.set_printoptions(threshold=16)
        dsc.txt("Inlines vector", self._ilines)
        dsc.txt("Xlines vector", self._xlines)
        dsc.txt("Time or depth slices vector", self.zslices)
        dsc.txt("Values", self._values.reshape(-1), self._values.dtype)
        np.set_printoptions(threshold=1000)
        dsc.txt(
            "Values, mean, stdev, minimum, maximum",
            self.values.mean(),
            self.values.std(),
            self.values.min(),
            self.values.max(),
        )
        dsc.txt("Trace ID codes", self._traceidcodes.reshape(-1))
        msize = float(self.values.size * 4) / (1024 * 1024 * 1024)
        dsc.txt("Minimum memory usage of array (GB)", msize)

        if flush:
            dsc.flush()
            return None

        return dsc.astext()

    # ==================================================================================
    # Copy, swapping, cropping, thinning...
    # ==================================================================================

    def copy(self):
        """Deep copy of a Cube() object to another instance.


        >>> mycube = xtgeo.cube_from_file(cube_dir + "/ib_test_cube2.segy")
        >>> mycube2 = mycube.copy()

        """
        xcube = Cube(
            ncol=self.ncol,
            nrow=self.nrow,
            nlay=self.nlay,
            xinc=self.xinc,
            yinc=self.yinc,
            zinc=self.zinc,
            xori=self.xori,
            yori=self.yori,
            zori=self.zori,
            yflip=self.yflip,
            segyfile=self.segyfile,
            rotation=self.rotation,
            values=self.values.copy(),
        )

        xcube.filesrc = self._filesrc

        xcube.ilines = self._ilines.copy()
        xcube.xlines = self._xlines.copy()
        xcube.traceidcodes = self._traceidcodes.copy()
        xcube.metadata.required = xcube

        return xcube

    def swapaxes(self):
        """Swap the axes inline vs xline, keep origin."""
        _cube_utils.swapaxes(self)

    def resample(self, incube, sampling="nearest", outside_value=None):
        """Resample a Cube object into this instance.

        Args:
            incube (Cube): A XTGeo Cube instance
            sampling (str): Sampling algorithm: 'nearest' for nearest node
                of 'trilinear' for trilinear interpoltion (more correct but
                slower)
            outside_value (None or float). If None, keep original, otherwise
                use this value

        Raises:
            ValueError: If cubes do not overlap

        Example:

            >>> import xtgeo
            >>> mycube1 = xtgeo.cube_from_file(cube_dir + "/ib_test_cube2.segy")
            >>> mycube2 = xtgeo.Cube(
            ...     xori=777574,
            ...     yori=6736507,
            ...     zori=1000,
            ...     xinc=10,
            ...     yinc=10,
            ...     zinc=4,
            ...     ncol=100,
            ...     nrow=100,
            ...     nlay=100,
            ...     yflip=mycube1.yflip,
            ...     rotation=mycube1.rotation
            ... )
            >>> mycube2.resample(mycube1)

        """
        _cube_utils.resample(
            self, incube, sampling=sampling, outside_value=outside_value
        )

    def do_thinning(self, icol, jrow, klay):
        """Thinning the cube by removing every N column, row and/or layer.

        Args:
            icol (int): Thinning factor for columns (usually inlines)
            jrow (int): Thinning factor for rows (usually xlines)
            klay (int): Thinning factor for layers

        Raises:
            ValueError: If icol, jrow or klay are out of reasonable range

        Example:

            >>> mycube1 = Cube(cube_dir + "/ib_test_cube2.segy")
            >>> mycube1.do_thinning(2, 2, 1)  # keep every second column, row
            >>> mycube1.to_file(outdir + '/mysegy_smaller.segy')

        """
        _cube_utils.thinning(self, icol, jrow, klay)

    def do_cropping(self, icols, jrows, klays, mode="edges"):
        """Cropping the cube by removing rows, columns, layers.

        Note that input boundary checking is currently lacking, and this
        is a currently a user responsibility!

        The 'mode' is used to determine to different 'approaches' on
        cropping. Examples for icols and mode 'edges':
        Here the tuple (N, M) will cut N first rows and M last rows.

        However, if mode is 'inclusive' then, it defines the range
        of rows to be included, and the numbering now shall be the
        INLINE, XLINE and DEPTH/TIME mode.

        Args:
            icols (int tuple): Cropping front, end of rows, or inclusive range
            jrows (int tuple): Cropping front, end of columns, or
                inclusive range
            klays (int tuple ): Cropping top, base layers, or inclusive range.
            mode (str): 'Default is 'edges'; alternative is 'inclusive'

        Example:
            Crop 10 columns from front, 2 from back, then 20 rows in front,
            40 in back, then no cropping of layers::

            >>> import xtgeo
            >>> mycube1 = xtgeo.cube_from_file(cube_dir + "/ib_test_cube2.segy")
            >>> mycube2 = mycube1.copy()
            >>> mycube1.do_cropping((10, 2), (20, 40), (0, 0))
            >>> mycube1.to_file(outdir + '/mysegy_smaller.segy')

            In stead, do cropping as 'inclusive' where inlines, xlines, slices
            arrays are known::

            >>> mycube2.do_cropping((11, 32), (112, 114), (150, 200))

        """
        useicols = icols
        usejrows = jrows
        useklays = klays

        if mode == "inclusive":
            # transfer to 'numbers to row/col/lay to remove' in front end ...
            useicols = (
                icols[0] - self._ilines[0],
                self._ilines[self._ncol - 1] - icols[1],
            )
            usejrows = (
                jrows[0] - self._xlines[0],
                self._xlines[self._nrow - 1] - jrows[1],
            )
            ntop = int((klays[0] - self.zori) / self.zinc)
            nbot = int((self.zori + self.nlay * self.zinc - klays[1] - 1) / (self.zinc))
            useklays = (ntop, nbot)

        logger.info(
            "Cropping at all cube sides: %s %s %s", useicols, usejrows, useklays
        )
        _cube_utils.cropping(self, useicols, usejrows, useklays)

    def values_dead_traces(self, newvalue):
        """Set values for traces flagged as dead.

        Dead traces have traceidcodes 2 and corresponding values in the cube
        will here receive a constant value to mimic "undefined".

        Args:
            newvalue (float): Set cube values to newvalues where traceid is 2.

        Return:
            oldvalue (float): The estimated simple 'average' of old value will
                be returned as (max + min)/2. If no dead traces, return None.
        """
        logger.info("Set values for dead traces, if any")

        if 2 in self._traceidcodes:
            minval = self._values[self._traceidcodes == 2].min()
            maxval = self._values[self._traceidcodes == 2].max()
            # a bit weird calculation of mean but kept for backward compatibility
            self._values[self._traceidcodes == 2] = newvalue
            return 0.5 * (minval + maxval)

        return None

    def get_xy_value_from_ij(self, iloc, jloc, ixline=False, zerobased=False):
        """Returns x, y coordinate from a single i j location.

        Args:
            iloc (int): I (col) location (base is 1)
            jloc (int): J (row) location (base is 1)
            ixline (bool): If True, then input locations are inline and xline position
            zerobased (bool): If True, first index is 0, else it is 1. This does not
                apply when ixline is set to True.

        Returns:
            The X, Y coordinate pair.
        """
        xval, yval = _cube_utils.get_xy_value_from_ij(
            self, iloc, jloc, ixline=ixline, zerobased=zerobased
        )
        return xval, yval

    def compute_attributes_in_window(
        self,
        upper: RegularSurface | float,
        lower: RegularSurface | float,
        ndiv: int = 10,
        interpolation: Literal["cubic", "linear"] = "cubic",
        minimum_thickness: float = 0.0,
    ) -> dict[RegularSurface]:
        """Return a cube's attributes as a set of surfaces, given two input surfaces.

        The attributes are computed vertically (per column) within a window defined by
        the two input surfaces and/or levels.

        The statistical measures can be min, max, mean, variance etc. A complete list of
        supported attributes is given below.

        * 'max' for maximum

        * 'min' for minimum

        * 'rms' for root mean square

        * 'mean' for expected value

        * 'var' for variance (population var; https://en.wikipedia.org/wiki/Variance)

        * 'maxpos' for maximum of positive values

        * 'maxneg' for negative maximum of negative values

        * 'maxabs' for maximum of absolute values

        * 'sumpos' for sum of positive values using cube sampling resolution

        * 'sumneg' for sum of negative values using cube sampling resolution

        * 'meanabs' for mean of absolute values

        * 'meanpos' for mean of positive values

        * 'meanneg' for mean of negative values

        * 'upper' will return a copy of the upper surface applied

        * 'lower' will return a copy of the lower surface applied


        Args:
            upper: The uppermost surface or constant level to compute within.
            lower: The lower surface or level to compute within.
            ndiv: Number of intervals for sampling within zrange. Default is 10.
                using 0.1 of cube Z increment as basis. A higher ndiv will increase
                CPU time and memory usage, but also increase the precision of the
                result.
            interpolation: 'cubic' or 'linear' for interpolation of the
                seismic signal, default here is 'cubic'.
            minimum_thickness: Minimum thickness (isochore or isochron) between the
                two surfaces. If the thickness is less or equal than this value,
                the result will be masked. Default is 0.0.

        Example::

            >>> import xtgeo
            >>> cube = xtgeo.cube_from_file("mycube.segy")
            >>> surf = xtgeo.surface_from_file("topreek.gri")
            >>> # sample in a total range of 30 m, 15 units above and 15 units below:
            >>> attrs = cube.compute_attributes_in_window((surf-15), (surf + 15))
            >>> attrs["max"].to_file("max.gri")  # save the 'max' attribute to file

        Note:
            This method is a significantly improved version of the
            :meth:`slice_cube_window` method within `RegularSurface()`, and it is
            strongly recommended to replace the former with this as soon as possible.

        .. versionadded:: 4.1

        """
        return _cube_window_attributes.CubeAttrs(
            self, upper, lower, ndiv, interpolation, minimum_thickness
        ).result()

    # =========================================================================
    # Cube extractions, e.g. XSection
    # =========================================================================

    def get_randomline(
        self,
        fencespec,
        zmin=None,
        zmax=None,
        zincrement=None,
        hincrement=None,
        atleast=5,
        nextend=2,
        sampling="nearest",
    ):
        """Get a randomline from a fence spesification.

        This randomline will be a 2D numpy with depth/time on the vertical
        axis, and length along as horizontal axis. Undefined values will have
        the np.nan value.

        The input fencespec is either a 2D numpy where each row is X, Y, Z, HLEN,
        where X, Y are UTM coordinates, Z is depth/time, and HLEN is a
        length along the fence, or a Polygons instance.

        If input fencspec is a numpy 2D, it is important that the HLEN array
        has a constant increment and ideally a sampling that is less than the
        Cube resolution. If a Polygons() instance, this is automated!

        Args:
            fencespec (:obj:`~numpy.ndarray` or :class:`~xtgeo.xyz.polygons.Polygons`):
                2D numpy with X, Y, Z, HLEN as rows or a xtgeo Polygons() object.
            zmin (float): Minimum Z (default is Cube Z minima/origin)
            zmax (float): Maximum Z (default is Cube Z maximum)
            zincrement (float): Sampling vertically, default is Cube ZINC/2
            hincrement (float or bool): Resampling horizontally. This applies only
                if the fencespec is a Polygons() instance. If None (default),
                the distance will be deduced automatically.
            atleast (int): Minimum number of horizontal samples (only if
                fencespec is a Polygons instance)
            nextend (int): Extend with nextend * hincrement in both ends (only if
                fencespec is a Polygons instance)
            sampling (str): Algorithm, 'nearest' or 'trilinear' (first is
                faster, second is more precise for continuous fields)

        Returns:
            A tuple: (hmin, hmax, vmin, vmax, ndarray2d)

        Raises:
            ValueError: Input fence is not according to spec.

        .. versionchanged:: 2.1 support for Polygons() as fencespec, and keywords
           hincrement, atleast and sampling

        .. seealso::
           Class :class:`~xtgeo.xyz.polygons.Polygons`
              The method :meth:`~xtgeo.xyz.polygons.Polygons.get_fence()` which can be
              used to pregenerate `fencespec`

        """
        if not isinstance(fencespec, (np.ndarray, Polygons)):
            raise ValueError(
                "fencespec must be a numpy or a Polygons() object. "
                f"Current type is {type(fencespec)}"
            )
        logger.info("Getting randomline...")
        res = _cube_utils.get_randomline(
            self,
            fencespec,
            zmin=zmin,
            zmax=zmax,
            zincrement=zincrement,
            hincrement=hincrement,
            atleast=atleast,
            nextend=nextend,
            sampling=sampling,
        )
        logger.info("Getting randomline... DONE")
        return res

    # =========================================================================
    # Import and export
    # =========================================================================

    @classmethod
    def _read_file(cls, sfile, fformat="guess"):
        """Import cube data from file.

        If fformat is not provided, the file type will be guessed based
        on file extension (e.g. segy og sgy for SEGY format)

        Args:
            sfile (str): Filename (as string or pathlib.Path instance).
            fformat (str): file format guess/segy/rms_regular/xtgregcube
                where 'guess' is default. Regard 'xtgrecube' format as experimental.
            deadtraces (float): Set 'dead' trace values to this value (SEGY
                only). Default is UNDEF value (a very large number).

        Raises:
            OSError: if the file cannot be read (e.g. not found)
            ValueError: Input is invalid

        Example::

            >>> zz = Cube()
            >>> zz.from_file(cube_dir + "/ib_test_cube2.segy")


        """
        mfile = FileWrapper(sfile)
        fmt = mfile.fileformat(fformat)
        kwargs = _data_reader_factory(fmt)(mfile)
        kwargs["filesrc"] = mfile.file
        return cls(**kwargs)

    def to_file(self, sfile, fformat="segy", pristine=False, engine=None):
        """Export cube data to file.

        Args:
            sfile (str): Filename
            fformat (str, optional): file format 'segy' (default) or
                'rms_regular'
            pristine (bool): If True, make SEGY from scratch.
            engine (str): Which "engine" to use.

        Example::
            >>> import xtgeo
            >>> zz = xtgeo.cube_from_file(cube_dir + "/ib_test_cube2.segy")
            >>> zz.to_file(outdir + '/some.rmsreg')
        """
        fobj = FileWrapper(sfile, mode="wb")

        fobj.check_folder(raiseerror=OSError)

        if engine is not None:
            warnings.warn(
                "Providing an 'engine' value is no longer supported and will have no "
                "effect in the future. segyio will eventually be used by default. "
                "Current default engine is 'xtgeo'.",
                UserWarning,
            )

        if fformat in FileFormat.SEGY.value:
            _cube_export.export_segy(self, fobj.name, pristine=pristine)
        elif fformat == "rms_regular":
            _cube_export.export_rmsreg(self, fobj.name)
        elif fformat == "xtgregcube":
            _cube_export.export_xtgregcube(self, fobj.name)
        else:
            extensions = FileFormat.extensions_string([FileFormat.SEGY])
            raise InvalidFileFormatError(
                f"File format {fformat} is invalid for type Cube. "
                f"Supported formats are {extensions}."
            )

    def to_roxar(
        self,
        project: Any,
        name: str,
        folder: str | None = None,
        propname: str = "seismic_attribute",
        domain: str = "time",
        compression: tuple[str, float] = ("wavelet", 5.0),
        target: str = "seismic",
    ):  # pragma: no cover
        """Export (transfer) a cube from a XTGeo cube object to Roxar data.

        Note:
            When project is file path (direct access, outside RMS) then
            ``to_roxar()`` will implicitly do a project save. Otherwise, the project
            will not be saved until the user do an explicit project save action.

        Args:
            project: Inside RMS use the magic 'project',
                else use path to RMS project, or a project reference
            name: Name of cube (seismic data) within RMS project.
            folder: Cubes may be stored under a folder in the tree, use '/'
                to seperate subfolders.
            propname: Name of grid property; only relevant when target is "grid" and
                defaults to "seismic_attribute"
            domain: 'time' (default) or 'depth'
            compression: Reference to Roxar API 'compression method' and 'compression
                tolerance', but implementation is pending. Hence inactive.
            target: Optionally, the seismic cube can be written to the `Grid model`
                tree in RMS. Internally, it will be convert to a "box" grid with one
                gridproperty, before it is written to RMS. The ``compression``and
                ``domain`` are not relevant when writing to grid model.

        Raises:
            To be described...

        Example::

            zz = xtgeo.cube_from_file('myfile.segy')
            zz.to_roxar(project, 'reek_cube')
            # write cube to "Grid model" tree in RMS instead
            zz.to_roxar(project, 'cube_as_grid', propname="impedance", target="grid")

        .. versionchanged:: 3.4 Add ``target`` and ``propname`` keys
        """

        if "grid" in target.lower():
            _tmpgrd = grid_from_cube(self, propname=name)
            _tmpprop = _tmpgrd.props[0]
            _tmpprop.name = propname if propname else "seismic_attribute"
            _tmpgrd.to_roxar(project, name)
            _tmpprop.to_roxar(project, name, _tmpprop.name)

        else:
            _cube_roxapi.export_cube_roxapi(
                self,
                project,
                name,
                folder=folder,
                domain=domain,
                compression=compression,
            )

    def _ensure_correct_values(
        self,
        values: None | bool | float | list | tuple | np.ndarray | np.ma.MaskedArray,
    ) -> np.ndarray:
        """Ensures that values is a 3D numpy (ncol, nrow, nlay), C order.

        Args:
            values: Values to process.

        """
        return_array = None
        if values is None or isinstance(values, bool):
            return_array = self._ensure_correct_values(0.0)

        elif isinstance(values, numbers.Number):
            array = np.zeros(self.dimensions, dtype=np.float32) + values
            return_array = array.astype(np.float32)  # ensure 32 bit floats

        elif isinstance(values, np.ndarray):
            # if the input is a maskedarray; need to convert and fill with zero
            if isinstance(values, np.ma.MaskedArray):
                warnings.warn(
                    "Input values is a masked numpy array, and masked nodes "
                    "will be set to zero in the cube instance.",
                    UserWarning,
                )
                values = np.ma.filled(values, fill_value=0)

            exp_len = np.prod(self.dimensions)
            if (
                values.size != exp_len
                or values.ndim not in (1, 3)
                or values.shape != self.dimensions
            ):
                raise ValueError(
                    "Input is of wrong shape or dimensions: "
                    f"{values.shape}, expected {self.dimensions}"
                    "or ({exp_len},)"
                )

            values = values.reshape(self.dimensions).astype(np.float32)

            if not values.flags.c_contiguous:
                values = np.ascontiguousarray(values)
            return_array = values

        elif isinstance(values, (list, tuple)):
            exp_len = int(np.prod(self.dimensions))
            if len(values) != exp_len:
                raise ValueError(
                    "The length of the input list or tuple is incorrect"
                    f"Input length is {len(values)} while expected length is {exp_len}"
                )

            return_array = np.array(values, dtype=np.float32).reshape(self.dimensions)

        else:
            raise ValueError(
                f"Cannot process _ensure_correct_values with input values: {values}"
            )

        if return_array is not None:
            return return_array

        raise RuntimeError("Unexpected error, return values are None")
