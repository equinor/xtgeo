# -*- coding: utf-8 -*-
"""The metadata module, currently experimental.

The metadata works through the various datatypes in XTGeo. For example::

    >>> surf = xtgeo.RegularSurface("somefile")
    >>> surf.metadata.required
    >>> ...
    >>> surf.metadata.optional.mean = surf.values.mean()

"""
# import datetime
from collections import OrderedDict

# from datetime import date
import xtgeo

xtg = xtgeo.common.XTGeoDialog()
logger = xtg.functionlogger(__name__)


class _OptionalMetaData:
    """Optional metadata are not required, but keys are limited.

    A limited sets of possible keys are available, and they can modified. This
    class can also have validation methods.
    """

    __slots__ = (
        "_name",
        "_shortname",
        "_datatype",
        "_md5sum",
        "_description",
        "_crs",
        "_datetime",
        "_deltadatetime",
        "_visuals",
        "_domain",
        "_user",
        "_field",
        "_source",
        "_modelid",
        "_ensembleid",
        "_units",
        "_mean",
        "_stddev",
        "_percentiles",
    )

    def __init__(self) -> None:
        self._name = "A Longer Descriptive Name e.g. from SMDA"
        self._shortname = "TheShortName"
        self._datatype = None
        self._md5sum = None
        self._description = "Some description"
        self._crs = None
        self._datetime = None
        self._deltadatetime = None
        self._visuals = {"colortable": "rainbow", "lower": None, "upper": None}
        self._domain = "depth"
        self._units = "metric"
        self._mean = None
        self._stddev = None
        self._percentiles = None
        self._user = "anonymous"
        self._field = "nofield"
        self._ensembleid = None
        self._modelid = None
        self._source = "unknown"

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, newname):
        # TODO: validation
        self._name = newname

    @property
    def datetime(self):
        return self._datetime

    @datetime.setter
    def datetime(self, newdate):
        # TODO: validation
        self._datetime = newdate

    @property
    def shortname(self):
        return self._shortname

    @shortname.setter
    def shortname(self, newname):
        if not isinstance(newname, str):
            raise ValueError("The shortname must be a string.")
        if len(newname) >= 32:
            raise ValueError("The shortname length must less or equal 32 letters.")

        self._shortname = newname

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, newstr):
        if not isinstance(newstr, str):
            raise ValueError("The description must be a string.")
        if len(newstr) >= 64:
            raise ValueError("The description length must less or equal 64 letters.")
        invalids = r"/$<>[]:\&%"
        if set(invalids).intersection(newstr):
            raise ValueError("The description constains invalid characters such as /.")

        self._description = newstr

    @property
    def md5sum(self):
        """Set or get the md5 checksum of file content.

        See generate_hash() method in e.g. RegularSurface.
        """
        return self._md5sum

    @md5sum.setter
    def md5sum(self, newhash):
        # TODO: validation
        self._md5sum = newhash

    def get_meta(self):
        """Return metadata as an OrderedDict."""
        meta = OrderedDict()
        for key in self.__slots__:
            newkey = key[1:]
            meta[newkey] = getattr(self, key)

        return meta


class MetaData:
    """Generic metadata class, not intended to be used directly."""

    def __init__(self):
        """Generic metadata class __init__, not be used directly."""
        self._required = OrderedDict()
        self._optional = _OptionalMetaData()
        self._freeform = OrderedDict()

        self._freeform = {"smda": "whatever"}

    def get_metadata(self):
        """Get all metadata that are present."""
        allmeta = OrderedDict()
        allmeta["_required_"] = self._required
        allmeta["_optional_"] = self._optional.get_meta()
        allmeta["_freeform_"] = self._freeform
        return allmeta

    @property
    def optional(self):
        """Return or set optional metadata.

        When setting optional names, it can be done in several ways...

        surf.metadata.optional.name = "New name"
        """
        # return a copy of the instance; the reason for this is to avoid manipulation
        # without validation
        return self._optional.get_meta()

    @optional.setter
    def optional(self, indict):
        # setting the optional key, including validation
        if not isinstance(indict, dict):
            raise ValueError(f"Input must be a dictionary, not a {type(indict)}")

        for key, value in indict.items():
            setattr(self._optional, "_" + key, value)

    @property
    def opt(self):
        """Return the metadata optional instance.

        This makes access to the _OptionalMetaData instance.

        Example::
            >>> surf = xtgeo.RegularSurface("somefile.gri")
            >>> surf.metadata.opt.shortname = "TopValysar"

        """
        return self._optional

    @optional.setter
    def optional(self, indict):
        # setting the optional key, including validation
        if not isinstance(indict, dict):
            raise ValueError(f"Input must be a dictionary, not a {type(indict)}")

        for key, value in indict.items():
            setattr(self._optional, "_" + key, value)

    @property
    def freeform(self):
        """Get or set the current freeform metadata dictionary."""
        return self._freeform

    @freeform.setter
    def freeform(self, adict):
        """Freeform is a whatever you want set, without any validation."""
        self._freeform = adict.copy()

    def generate_fmu_name(self):
        """Generate FMU name on form xxxx--yyyy--date but no suffix."""
        fname = ""
        first = "prefix"
        fname += first
        fname += "--"
        fname += self._optional._shortname.lower()
        if self._optional._datetime:
            fname += "--"
            fname += str(self._optional._datetime)
        return fname


class MetaDataRegularSurface(MetaData):
    """Metadata for RegularSurface() objects."""

    REQUIRED = OrderedDict(
        [
            ("ncol", 1),
            ("nrow", 1),
            ("xori", 0.0),
            ("yori", 0.0),
            ("xinc", 1.0),
            ("yinc", 1.0),
            ("yflip", 1),
            ("rotation", 0.0),
            ("undef", xtgeo.UNDEF),
        ]
    )

    def __init__(self):
        """Docstring."""
        super().__init__()
        self._required = __class__.REQUIRED
        self._optional._datatype = "Regular Surface"

    @property
    def required(self):
        """Get of set required metadata."""
        return self._required

    @required.setter
    def required(self, obj):
        if not isinstance(obj, xtgeo.RegularSurface):
            raise ValueError("Input object is not a RegularSurface()")

        self._required["ncol"] = obj.ncol
        self._required["nrow"] = obj.nrow
        self._required["xori"] = obj.xori
        self._required["yori"] = obj.yori
        self._required["xinc"] = obj.xinc
        self._required["yinc"] = obj.yinc
        self._required["yflip"] = obj.yflip
        self._required["rotation"] = obj.rotation
        self._required["undef"] = obj.undef


class MetaDataRegularCube(MetaData):
    """Metadata for Cube() objects."""

    # allowed optional keys; these are set to avoid discussions
    REQUIRED = OrderedDict(
        [
            ("ncol", 1),
            ("nrow", 1),
            ("nlay", 1),
            ("xori", 0.0),
            ("yori", 0.0),
            ("zori", 0.0),
            ("xinc", 1.0),
            ("yinc", 1.0),
            ("zinc", 1.0),
            ("yflip", 1),
            ("zflip", 1),
            ("rotation", 0.0),
            ("undef", xtgeo.UNDEF),
        ]
    )

    def __init__(self):
        """Docstring."""
        super().__init__()
        self._required = __class__.REQUIRED
        self._optional._datatype = "Regular Cube"

    @property
    def required(self):
        """Get of set required metadata."""
        return self._required

    @required.setter
    def required(self, obj):
        if not isinstance(obj, xtgeo.Cube):
            raise ValueError("Input object is not a regular Cube()")

        self._required["ncol"] = obj.ncol
        self._required["nrow"] = obj.nrow
        self._required["nlay"] = obj.nlay
        self._required["xori"] = obj.xori
        self._required["yori"] = obj.yori
        self._required["zori"] = obj.zori
        self._required["xinc"] = obj.xinc
        self._required["yinc"] = obj.yinc
        self._required["zinc"] = obj.zinc
        self._required["yflip"] = obj.yflip
        self._required["zflip"] = 1
        self._required["rotation"] = obj.rotation
        self._required["undef"] = obj.undef


class MetaDataCPGeometry(MetaData):
    """Metadata for Grid() objects of type simplified CornerPoint Geometry."""

    REQUIRED = OrderedDict(
        [
            ("ncol", 1),
            ("nrow", 1),
            ("nlay", 1),
            ("xshift", 0.0),
            ("yshift", 0.0),
            ("zshift", 0.0),
            ("xscale", 1.0),
            ("yscale", 1.0),
            ("zscale", 1.0),
        ]
    )

    def __init__(self):
        """Docstring."""
        super().__init__()
        self._required = __class__.REQUIRED
        self._optional._datatype = "CornerPoint GridGeometry"

    @property
    def required(self):
        """Get of set required metadata."""
        return self._required

    @required.setter
    def required(self, obj):
        if not isinstance(obj, xtgeo.Grid):
            raise ValueError("Input object is not a Grid()")

        self._required["ncol"] = obj.ncol
        self._required["nrow"] = obj.nrow
        self._required["nlay"] = obj.nlay
        self._required["xshift"] = 0.0  # hardcoded so far
        self._required["yshift"] = 0.0
        self._required["zshift"] = 0.0
        self._required["xscale"] = 1.0
        self._required["yscale"] = 1.0
        self._required["zscale"] = 1.0
        self._required["subgrids"] = obj.get_subgrids()


class MetaDataCPProperty(MetaData):
    """Metadata for GridProperty() objects belonging to CPGeometry."""

    REQUIRED = OrderedDict(
        [
            ("ncol", 1),
            ("nrow", 1),
            ("nlay", 1),
            ("codes", None),
            ("discrete", False),
        ]
    )

    def __init__(self):
        """Docstring."""
        super().__init__()
        self._required = __class__.REQUIRED
        self._optional._datatype = "CornerPoint GridProperty"

    @property
    def required(self):
        """Get of set required metadata."""
        return self._required

    @required.setter
    def required(self, obj):
        if not isinstance(obj, xtgeo.GridProperty):
            raise ValueError("Input object is not a GridProperty()")

        self._required["ncol"] = obj.ncol
        self._required["nrow"] = obj.nrow
        self._required["nlay"] = obj.nlay
        self._required["codes"] = obj.codes
        self._required["discrete"] = obj.isdiscrete


class MetaDataWell(MetaData):
    """Metadata for single Well() objects."""

    REQUIRED = OrderedDict(
        [
            ("rkb", 0.0),
            ("xpos", 0.0),
            ("ypos", 0.0),
            ("name", "noname"),
            ("wlogs", dict()),
            ("mdlogname", None),
            ("zonelogname", None),
        ]
    )

    def __init__(self):
        """Initialisation for Well metadata."""
        super().__init__()
        self._required = __class__.REQUIRED
        self._optional._datatype = "Well"

    @property
    def required(self):
        """Get of set required metadata."""
        return self._required

    @required.setter
    def required(self, obj):
        if not isinstance(obj, xtgeo.Well):
            raise ValueError("Input object is not a Well() instance!")

        self._required["rkb"] = obj.rkb
        self._required["xpos"] = obj.xpos
        self._required["ypos"] = obj.ypos
        self._required["name"] = obj.wname
        self._required["wlogs"] = obj.get_wlogs()
        self._required["mdlogname"] = obj.mdlogname
        self._required["zonelogname"] = obj.zonelogname
