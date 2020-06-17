# coding: utf-8
"""Module for a seismic (or whatever) cube."""
from __future__ import print_function, division

import os.path
import tempfile
import sys

import numpy as np

import xtgeo
from xtgeo.common import XTGeoDialog
from xtgeo.common import XTGDescription
import xtgeo.common.sys as xtgeosys

from xtgeo.cube import _cube_import
from xtgeo.cube import _cube_export
from xtgeo.cube import _cube_utils
from xtgeo.cube import _cube_roxapi


xtg = XTGeoDialog()
logger = xtg.functionlogger(__name__)


# =============================================================================
# METHODS as wrappers to class init + import


def cube_from_file(mfile, fformat="guess"):
    """This makes an instance of a Cube directly from file import.

    Args:
        mfile (str): Name of file
        fformat (str): See :meth:`Cube.from_file`

    Example::

        import xtgeo
        mycube = xtgeo.cube_from_file('some_cube.segy')
    """

    obj = Cube()

    obj.from_file(mfile, fformat=fformat)

    return obj


def cube_from_roxar(project, name, folder=None):
    """This makes an instance of a Cube directly from roxar input.

    The folder is a string on form 'a' or 'a/b' if subfolders are present

    Example::

        import xtgeo
        mycube = xtgeo.cube_from_roxar(project, 'DepthCube')

    """

    obj = Cube()

    obj.from_roxar(project, name, folder=folder)

    return obj


class Cube(object):  # pylint: disable=too-many-public-methods
    """Class for a (seismic) cube in the XTGeo framework.

    The values are a numpy array, 3D Float (4 bytes; float32). The
    array is (ncol, nrow, nlay) regular 3D numpy,
    with internal C ordering (nlay fastest).

    The cube object instance can be initialized by either a
    spesification, or via a file import. The import is most
    common, and usually SEGY, but also other formats are
    available (or will be).

    Examples::

        from xtgeo.cube import Cube

        # a user defined cube:
        vals = np.zeros((40, 30, 10), dtype=np.float32)

        mycube = Cube(xori=100.0, yori=200.0, ncol=40, nrow=30,
                      nlay=10, rotation=30, values=vals)

        # or from a file
        mycube = Cube('somefile.segy')

    """

    def __init__(self, *args, **kwargs):
        """Initiate a Cube instance."""

        self._filesrc = None
        self._segyfile = None
        self._ilines = None
        self._xlines = None
        self._xori = 0.0
        self._yori = 0.0
        self._zori = 0.0
        self._ncol = 5
        self._nrow = 3
        self._nlay = 4
        self._xinc = 25.0
        self._yinc = 25.0
        self._zinc = 2.0
        self._yflip = 1
        self._values = np.zeros((5, 3, 4), dtype=np.float32)
        self._ilines = np.array(range(1, 5 + 1), dtype=np.int32)
        self._xlines = np.array(range(1, 3 + 1), dtype=np.int32)
        self._rotation = 0.0
        self._traceidcodes = np.ones((5, 3), dtype=np.int32)
        self._undef = xtgeo.UNDEF
        self._undef_limit = xtgeo.UNDEF_LIMIT

        if len(args) >= 1:
            fformat = kwargs.get("fformat", "guess")
            self.from_file(args[0], fformat=fformat)
        else:
            self._filesrc = None
            self._xori = kwargs.get("xori", 0.0)
            self._yori = kwargs.get("yori", 0.0)
            self._zori = kwargs.get("zori", 0.0)
            self._ncol = kwargs.get("ncol", 5)
            self._nrow = kwargs.get("nrow", 3)
            self._nlay = kwargs.get("nlay", 2)
            self._xinc = kwargs.get("xinc", 25.0)
            self._yinc = kwargs.get("yinc", 25.0)
            self._zinc = kwargs.get("zinc", 2.0)
            self._yflip = kwargs.get("yflip", 1)
            self._values = kwargs.get("values", None)
            self._rotation = kwargs.get("rotation", 0.0)
            if self._values is None:
                vals = np.zeros((self._ncol, self._nrow, self._nlay), dtype=np.float32)
                self._values = vals
                self._ilines = np.array(range(1, self._ncol + 1), dtype=np.int32)
                self._xlines = np.array(range(1, self._nrow + 1), dtype=np.int32)
                self._traceidcodes = np.ones((self._nrow, self._nrow), dtype=np.int32)

            self._segyfile = kwargs.get("segyfile", None)

    def __repr__(self):
        avg = self.values.mean()
        dsc = (
            "{0.__class__} (ncol={0.ncol!r}, "
            "nrow={0.nrow!r}, nlay={0.nlay!r}, "
            "original file: {0._filesrc}), "
            "average {1}, ID=<{2}>".format(self, avg, id(self))
        )
        return dsc

    def __str__(self):
        return self.describe(flush=False)

    # =========================================================================
    # Get and Set properties (tend to pythonic properties rather than
    # javaic get & set syntax)
    # =========================================================================

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
    def xori(self):
        """The XORI (origin corner) coordinate."""
        return self._xori

    @xori.setter
    def xori(self, val):
        logger.warning("Changing xori is risky")
        self._xori = val

    @property
    def yori(self):
        """The YORI (origin corner) coordinate."""
        return self._yori

    @yori.setter
    def yori(self, val):
        logger.warning("Changing yori is risky")
        self._yori = val

    @property
    def zori(self):
        """The ZORI (origin corner) coordinate."""
        return self._zori

    @zori.setter
    def zori(self, val):
        logger.warning("Changing zori is risky")
        self._zori = val

    @property
    def xinc(self):
        """The XINC (increment X) as property."""
        return self._xinc

    @xinc.setter
    def xinc(self, val):
        logger.warning("Changing xinc is risky")
        self._xinc = val

    @property
    def yinc(self):
        """The YINC (increment Y)."""
        return self._yinc

    @yinc.setter
    def yinc(self, val):
        logger.warning("Changing yinc is risky")
        self._yinc = val

    @property
    def zinc(self):
        """ The ZINC (increment Z)."""
        return self._zinc

    @zinc.setter
    def zinc(self, val):
        logger.warning("Changing zinc is risky")
        self._zinc = val

    @property
    def rotation(self):
        """The rotation (columns vector, anticlock from X axis in degrees)."""
        return self._rotation

    @rotation.setter
    def rotation(self, val):
        logger.warning("Changing rotation is risky")
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
        # This is a derived property
        zslices = range(
            int(self.zori), int(self.zori + self.nlay * self.zinc), int(self.zinc)
        )
        zslices = np.array(zslices)
        return zslices

    @property
    def traceidcodes(self):
        """The trace identifaction codes array (ncol, nrow)."""
        return self._traceidcodes

    @traceidcodes.setter
    def traceidcodes(self, values):
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

        if not isinstance(values, np.ndarray):
            raise ValueError("Input is not a numpy array")

        vshape = values.shape

        if vshape != (self._ncol, self._nrow, self._nlay):
            raise ValueError("Wrong dimensions of input numpy")

        values = np.ascontiguousarray(values, dtype=np.float32)

        self._values = values

    # =========================================================================
    # Describe
    # =========================================================================
    def describe(self, flush=True):
        """Describe an instance by printing to stdout or return"""

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

    # =========================================================================
    # Copy, swapping, cropping, thinning...
    # =========================================================================

    def copy(self):
        """Copy a xtgeo.cube.Cube object to another instance::

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

            >>> mycube1 = Cube('mysegyfile.segy')
            >>> mycube2 = Cube(xori=461500, yori=5926800, zori=1550,
                   xinc=40, yinc=40, zinc=5, ncol=200, nrow=100,
                   nlay=100, rotation=mycube1.rotation)
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

            >>> mycube1 = Cube('mysegyfile.segy')
            >>> mycube1.do_thinning(2, 2, 1)  # keep every second column, row
            >>> mycube1.to_file('mysegy_smaller.segy')

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

            >>> mycube1 = Cube('mysegyfile.segy')
            >>> mycube2 = mycube1.copy()
            >>> mycube1.do_cropping((10, 2), (20, 40), (0, 0))
            >>> mycube1.to_file('mysegy_smaller.segy')

            In stead, do cropping as 'inclusive' where inlines, xlines, slices
            arrays are known::

            >>> mycube2.do_cropping((112, 327), (1120, 1140), (1500, 2000))

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
        """Set values for traces flagged as dead, i.e. have traceidcodes 2,
        and return the (average of) old values.

        Args:
            newvalue (float):
        Return:
            oldvalue (float): The estimated simple average of old value will
                be returned. If no dead traces, then None will be returned.
        """

        try:
            logger.info(self._values.shape)
            logger.info("%s %s", self._traceidcodes.shape, self._traceidcodes.dtype)
            minval = self._values[self._traceidcodes == 2].min()
            maxval = self._values[self._traceidcodes == 2].max()
            logger.info("MIN MAX %s %s", minval, maxval)
            avgold = 0.5 * (minval + maxval)
            self._values[self._traceidcodes == 2] = newvalue
            # self._values = np.where(
            #     self._traceidcodes == 2, newvalue, self._values
            # )
            logger.info("Setting dead traces done")
        except ValueError:
            avgold = None
            logger.info("No dead traces")

        return avgold

    def get_xy_value_from_ij(self, iloc, jloc, ixline=False, zerobased=False):
        """Returns x, y from a single i j location.

        Args:
            iloc (int): I (col) location (base is 1)
            jloc (int): J (row) location (base is 1)
            ixline (bool): If True, then input locations are inline and xline position
            zerobased (bool): If True, first index is 0, else it is 1. This does not
                apply when ixline is set to True.
        Returns:
            The z value at location iloc, jloc, None if undefined cell.
        """

        xval, yval = _cube_utils.get_xy_value_from_ij(
            self, iloc, jloc, ixline=ixline, zerobased=zerobased
        )

        return xval, yval

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

        .. versionchanged:: 2.1.0 support for Polygons() as fencespec, and keywords
           hincrement, atleast and sampling

        .. seealso::
           Class :class:`~xtgeo.xyz.polygons.Polygons`
              The method :meth:`~xtgeo.xyz.polygons.Polygons.get_fence()` which can be
              used to pregenerate `fencespec`

        """
        if not isinstance(fencespec, (np.ndarray, xtgeo.Polygons)):
            raise ValueError(
                "fencespec must be a numpy or a Polygons() object. "
                "Current type is {}".format(type(fencespec))
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

    def from_file(self, sfile, fformat="guess", engine="segyio"):
        """Import cube data from file.

        If fformat is not provided, the file type will be guessed based
        on file extension (e.g. segy og sgy for SEGY format)

        Args:
            sfile (str): Filename (as string or pathlib.Path)
            fformat (str): file format guess/segy/rms_regular
                where 'guess' is default
            engine (str): For the SEGY reader, 'xtgeo' is builtin
                while 'segyio' uses the SEGYIO library (default)
            deadtraces (float): Set 'dead' trace values to this value (SEGY
                only). Default is UNDEF value (a very large number)

        Raises:
            OSError if the file cannot be read (e.g. not found)

        Example::

            >>> zz = Cube()
            >>> zz.from_file('some.segy')


        """
        fobj = xtgeosys._XTGeoFile(sfile)
        fobj.check_file(raiseerror=OSError)

        _froot, fext = fobj.splitext(lower=True)

        if fformat == "guess":
            if not fext:
                logger.critical("File extension missing. STOP")
                sys.exit(9)
            else:
                fformat = fext.lower()

        if "rms" in fformat:
            _cube_import.import_rmsregular(self, fobj.name)
        elif fformat in ("segy", "sgy"):
            _cube_import.import_segy(self, fobj.name, engine=engine)
        elif fformat == "storm":
            _cube_import.import_stormcube(self, fobj.name)
        else:
            logger.error("Invalid or unknown file format")

        self._filesrc = fobj.name

    def to_file(self, sfile, fformat="segy", pristine=False, engine="xtgeo"):
        """Export cube data to file.

        Args:
            sfile (str): Filename
            fformat (str, optional): file format 'segy' (default) or
                'rms_regular'
            pristine (bool): If True, make SEGY from scratch.
            engine (str): Which "engine" to use.

        Example::
            >>> zz = Cube('some.segy')
            >>> zz.to_file('some.rmsreg')
        """
        fobj = xtgeosys._XTGeoFile(sfile, mode="wb")

        fobj.check_folder(raiseerror=OSError)

        if fformat == "segy":
            _cube_export.export_segy(self, fobj.name, pristine=pristine, engine=engine)
        elif fformat == "rms_regular":
            _cube_export.export_rmsreg(self, fobj.name)
        else:
            logger.error("Invalid file format")

    def from_roxar(self, project, name, folder=None):  # pragma: no cover
        """Import (transfer) a cube from a Roxar seismic object to XTGeo.

        Args:
            project (str): Inside RMS use the magic 'project', else use
                path to RMS project
            name (str): Name of cube within RMS project.
            folder (str): Folder name in in RMS if present; use '/' to seperate
                subfolders

        Raises:
            To be described...

        Example::

            zz = Cube()
            zz.from_roxar(project, 'truth_reek_seismic_depth_2000', folder="alt/depth")

        """
        _cube_roxapi.import_cube_roxapi(self, project, name, folder=folder)

    def to_roxar(
        self, project, name, folder=None, domain="time", compression=("wavelet", 5)
    ):  # pragma: no cover
        """Export (transfer) a cube from a XTGeo cube object to Roxar data.

        Args:
            project (str): Inside RMS use the magic 'project', else use
                path to RMS project
            name (str): Name of cube (seismic data) within RMS project.
            folder (str): Cubes may be stored under a folder in the tree, use '/'
                to seperate subfolders.
            domain (str): 'time' (default) or 'depth'
            compression (tuple): description to come...

        Raises:
            To be described...

        Example::

            zz = xtgeo.cube.Cube('myfile.segy')
            zz.to_roxar(project, 'reek_cube')

            # alternative
            zz = xtgeo.cube_from_file('myfile.segy')
            zz.to_roxar(project, 'reek_cube')

        """
        _cube_roxapi.export_cube_roxapi(
            self, project, name, folder=folder, domain=domain, compression=compression
        )

    @staticmethod
    def scan_segy_traces(sfile, outfile=None):
        """Scan a SEGY file traces and print limits info to STDOUT or file.

        Args:
            sfile (str): Name of SEGY file
            outfile: File where store scanned trace info, if empty or None
                output goes to STDOUT.
        """

        oflag = False
        # if outfile is none, it means that you want to plot on STDOUT
        if outfile is None:
            fx = tempfile.NamedTemporaryFile(delete=False, prefix="tmpxgeo")
            fx.close()
            outfile = fx.name
            logger.info("TMP file name is %s", outfile)
            oflag = True

        _cube_import._import_segy_xtgeo(
            sfile, scanheadermode=False, scantracemode=True, outfile=outfile
        )

        if oflag:
            # pass
            logger.info("OUTPUT to screen...")
            with open(outfile, "r") as out:
                for line in out:
                    print(line.rstrip("\r\n"))
            os.remove(outfile)

    @staticmethod
    def scan_segy_header(sfile, outfile=None):
        """Scan a SEGY file header and print info to screen or file.

        Args:
            sfile (str): Name of SEGY file
            outfile (str): File where store header info, if empty or None
                output goes to screen (STDOUT).
        """

        flag = False
        # if outfile is none, it means that you want to print on STDOUT
        if outfile is None:
            fc = tempfile.NamedTemporaryFile(delete=False, prefix="tmpxtgeo")
            fc.close()
            outfile = fc.name
            logger.info("TMP file name is %s", outfile)
            flag = True

        _cube_import._import_segy_xtgeo(
            sfile, scanheadermode=True, scantracemode=False, outfile=outfile
        )

        if flag:
            logger.info("OUTPUT to screen...")
            with open(outfile, "r") as out:
                for line in out:
                    print(line.rstrip("\r\n"))
            os.remove(outfile)
