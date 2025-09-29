"""The metadata module, currently experimental.

The metadata works through the various datatypes in XTGeo. For example::

    >>> import xtgeo
    >>> surf = xtgeo.surface_from_file(surface_dir + "/topreek_rota.gri")
    >>> surf.metadata.required
    dict([('ncol', 554),...
    >>> surf.metadata.optional.mean = surf.values.mean()

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import xtgeo
from xtgeo.common.constants import UNDEF
from xtgeo.common.log import null_logger

if TYPE_CHECKING:
    from datetime import datetime

    from xtgeo.cube.cube1 import Cube
    from xtgeo.grid3d.grid import Grid, GridProperty
    from xtgeo.surface.regular_surface import RegularSurface
    from xtgeo.well.well1 import Well


logger = null_logger(__name__)


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
        self._datatype: str | None = None
        self._md5sum: str | None = None
        self._description = "Some description"
        self._crs = None
        self._datetime: datetime | str | None = None
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
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, newname: str) -> None:
        # TODO: validation
        self._name = newname

    @property
    def datetime(self) -> datetime | str | None:
        return self._datetime

    @datetime.setter
    def datetime(self, newdate: datetime | str) -> None:
        # TODO: validation
        self._datetime = newdate

    @property
    def shortname(self) -> str:
        return self._shortname

    @shortname.setter
    def shortname(self, newname: str) -> None:
        if not isinstance(newname, str):
            raise ValueError("The shortname must be a string.")
        if len(newname) >= 32:
            raise ValueError("The shortname length must less or equal 32 letters.")

        self._shortname = newname

    @property
    def description(self) -> str:
        return self._description

    @description.setter
    def description(self, newstr: str) -> None:
        if not isinstance(newstr, str):
            raise ValueError("The description must be a string.")
        if len(newstr) >= 64:
            raise ValueError("The description length must less or equal 64 letters.")
        invalids = r"/$<>[]:\&%"
        if set(invalids).intersection(newstr):
            raise ValueError("The description contains invalid characters such as /.")

        self._description = newstr

    @property
    def md5sum(self) -> str | None:
        """Set or get the md5 checksum of file content.

        See generate_hash() method in e.g. RegularSurface.
        """
        return self._md5sum

    @md5sum.setter
    def md5sum(self, newhash: str) -> None:
        # TODO: validation
        self._md5sum = newhash

    def get_meta(self) -> dict[str, Any]:
        """Return metadata as an dict."""
        meta = {}
        for key in self.__slots__:
            newkey = key[1:]
            meta[newkey] = getattr(self, key)

        return meta


class MetaData:
    """Generic metadata class, not intended to be used directly."""

    def __init__(self) -> None:
        """Generic metadata class __init__, not be used directly."""
        self._required: dict[str, Any] = {}
        self._optional = _OptionalMetaData()
        self._freeform = {}

        self._freeform = {"smda": "whatever"}

    def get_metadata(self) -> dict[str, Any]:
        """Get all metadata that are present."""
        allmeta = {}
        allmeta["_required_"] = self._required
        allmeta["_optional_"] = self._optional.get_meta()
        allmeta["_freeform_"] = self._freeform
        return allmeta

    @property
    def optional(self) -> dict[str, Any]:
        """Return or set optional metadata.

        When setting optional names, it can be done in several ways...

        surf.metadata.optional.name = "New name"
        """
        # return a copy of the instance; the reason for this is to avoid manipulation
        # without validation
        return self._optional.get_meta()

    @optional.setter
    def optional(self, indict: dict[str, Any]) -> None:
        # setting the optional key, including validation
        if not isinstance(indict, dict):
            raise ValueError(f"Input must be a dictionary, not a {type(indict)}")

        for key, value in indict.items():
            setattr(self._optional, "_" + key, value)

    @property
    def opt(self) -> _OptionalMetaData:
        """Return the metadata optional instance.

        This makes access to the _OptionalMetaData instance.

        Example::
            >>> import xtgeo
            >>> surf = xtgeo.surface_from_file(surface_dir + "/topreek_rota.gri")
            >>> surf.metadata.opt.shortname = "TopValysar"

        """
        return self._optional

    @property
    def freeform(self) -> dict[str, Any]:
        """Get or set the current freeform metadata dictionary."""
        return self._freeform

    @freeform.setter
    def freeform(self, adict: dict[str, Any]) -> None:
        """Freeform is a whatever you want set, without any validation."""
        self._freeform = adict.copy()

    def generate_fmu_name(self) -> str:
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

    REQUIRED: dict[str, Any] = {
        "ncol": 1,
        "nrow": 1,
        "xori": 0.0,
        "yori": 0.0,
        "xinc": 1.0,
        "yinc": 1.0,
        "yflip": 1,
        "rotation": 0.0,
        "undef": UNDEF,
    }

    def __init__(self) -> None:
        """Docstring."""
        super().__init__()
        self._required = self.REQUIRED
        self._optional._datatype = "Regular Surface"

    @property
    def required(self) -> dict[str, Any]:
        """Get of set required metadata."""
        return self._required

    @required.setter
    def required(self, obj: RegularSurface) -> None:
        if not isinstance(obj, xtgeo.RegularSurface):  # type: ignore[attr-defined]
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
    REQUIRED: dict[str, Any] = {
        "ncol": 1,
        "nrow": 1,
        "nlay": 1,
        "xori": 0.0,
        "yori": 0.0,
        "zori": 0.0,
        "xinc": 1.0,
        "yinc": 1.0,
        "zinc": 1.0,
        "yflip": 1,
        "zflip": 1,
        "rotation": 0.0,
        "undef": UNDEF,
    }

    def __init__(self) -> None:
        """Docstring."""
        super().__init__()
        self._required = self.REQUIRED
        self._optional._datatype = "Regular Cube"

    @property
    def required(self) -> dict[str, Any]:
        """Get of set required metadata."""
        return self._required

    @required.setter
    def required(self, obj: Cube) -> None:
        if not isinstance(obj, xtgeo.Cube):  # type: ignore[attr-defined]
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

    REQUIRED: dict[str, Any] = {
        "ncol": 1,
        "nrow": 1,
        "nlay": 1,
        "xshift": 0.0,
        "yshift": 0.0,
        "zshift": 0.0,
        "xscale": 1.0,
        "yscale": 1.0,
        "zscale": 1.0,
    }

    def __init__(self) -> None:
        """Docstring."""
        super().__init__()
        self._required = self.REQUIRED
        self._optional._datatype = "CornerPoint GridGeometry"

    @property
    def required(self) -> dict[str, Any]:
        """Get of set required metadata."""
        return self._required

    @required.setter
    def required(self, obj: Grid) -> None:
        if not isinstance(obj, xtgeo.Grid):  # type: ignore[attr-defined]
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

    REQUIRED: dict[str, Any] = {
        "ncol": 1,
        "nrow": 1,
        "nlay": 1,
        "codes": None,
        "discrete": False,
    }

    def __init__(self) -> None:
        """Docstring."""
        super().__init__()
        self._required = self.REQUIRED
        self._optional._datatype = "CornerPoint GridProperty"

    @property
    def required(self) -> dict[str, Any]:
        """Get of set required metadata."""
        return self._required

    @required.setter
    def required(self, obj: GridProperty) -> None:
        if not isinstance(obj, xtgeo.GridProperty):  # type: ignore[attr-defined]
            raise ValueError("Input object is not a GridProperty()")

        self._required["ncol"] = obj.ncol
        self._required["nrow"] = obj.nrow
        self._required["nlay"] = obj.nlay
        self._required["codes"] = obj.codes
        self._required["discrete"] = obj.isdiscrete


class MetaDataWell(MetaData):
    """Metadata for single Well() objects."""

    REQUIRED: dict[str, Any] = {
        "rkb": 0.0,
        "xpos": 0.0,
        "ypos": 0.0,
        "name": "noname",
        "wlogs": {},
        "mdlogname": None,
        "zonelogname": None,
    }

    def __init__(self) -> None:
        """Initialisation for Well metadata."""
        super().__init__()
        self._required = self.REQUIRED
        self._optional._datatype = "Well"

    @property
    def required(self) -> dict[str, Any]:
        """Get of set required metadata."""
        return self._required

    @required.setter
    def required(self, obj: Well) -> None:
        if not isinstance(obj, xtgeo.Well):  # type: ignore[attr-defined]
            raise ValueError("Input object is not a Well() instance!")

        self._required["rkb"] = obj.rkb
        self._required["xpos"] = obj.xpos
        self._required["ypos"] = obj.ypos
        self._required["name"] = obj.wname
        self._required["wlogs"] = obj.get_wlogs()
        self._required["mdlogname"] = obj.mdlogname
        self._required["zonelogname"] = obj.zonelogname
