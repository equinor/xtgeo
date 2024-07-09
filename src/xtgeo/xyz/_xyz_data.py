"""Module for private _XYZData class.

Note that that the design of this targets Well and general XYZ data (Points/Polygons),
hence the intentions is to let this work as a general 'engine' for dataframe'ish data
in xtgeo, at least Well, Points, Polygons. (But in the first round, it is implemented
for Wells only). Dataframes looks like:

           X_UTME       Y_UTMN    Z_TVDSS     MDepth    PHIT      KLOGH      Sw
0      463256.911  5930542.294   -49.0000     0.0000     NaN        NaN     NaN ...
1      463256.912  5930542.295   -48.2859     0.5000     NaN        NaN     NaN ...
2      463256.913  5930542.296   -47.5735     1.0000     NaN        NaN     NaN ...
3      463256.914  5930542.299   -46.8626     1.5000     NaN        NaN     NaN ...
4      463256.916  5930542.302   -46.1533     2.0000     NaN        NaN     NaN ...
              ...          ...        ...        ...     ...        ...     ...

Where each attr (log) has a attr_types dictionary, telling if the columns are treated
as discrete (DISC) or continuous (CONT). In addition there is a attr_records
dict, storing the unit+scale for continuous logs/attr (defaulted to tuple ("", "")) or a
dictionary of codes (defaulted to {}, if the column if DISC type (this is optional,
and perhaps only relevant for Well data).

The 3 first columns are the XYZ coordinates or XY coordinates + value:
X, Y, Z or X, Y, V. An optional fourth column as also possible as polygon_id.
All the rest are free 'attributes', which for wells will be well logs. Hence:

    attr_types ~ refer to attr_types for XYZ and Well data
    attr_records ~ refer to attr_records for Well data and possibly Points/Polygons

If a column is added to the dataframe, then the methods here will try to guess the
attr_type and attr_record, and add those; similarly of a column is removed, the
corresponding entries in attr_types and attr_records will be deleted.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from joblib import hash as jhash

from xtgeo import _cxtgeo
from xtgeo._cxtgeo import XTGeoCLibError
from xtgeo.common._xyz_enum import _AttrName, _AttrType, _XYZType
from xtgeo.common.constants import UNDEF_CONT, UNDEF_DISC
from xtgeo.common.log import null_logger
from xtgeo.common.sys import _convert_carr_double_np, _get_carray

if TYPE_CHECKING:
    from collections.abc import Sequence


logger = null_logger(__name__)


CONT_DEFAULT_RECORD = ("", "")  # unit and scale, where emptry string indicates ~unknown


class _XYZData:
    """Private class for the XYZ and Well log data, where a Pandas dataframe is core.

    The data are stored in pandas dataframes, and by default, all columns are float, and
    np.nan defines undefined values. Even if they are DISC. The reason for this is
    restrictions in older versions of Pandas.

    All values in the dataframe shall be numbers.

    The attr_types is on form {"PHIT": CONT, "FACIES": DISC, ...}

    The attr_records is somewhat heterogeneous, on form:
    {"PHIT": ("unit", "scale"), "FACIES": {0:BG, 2: "SST", 4: "CALC"}}
    Hence the CONT logs hold a tuple or list with 2 str members, or None, while DISC
    log holds a dict where the key is an int and the value is a string.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        attr_types: dict[str, str] | None = None,
        attr_records: dict[str, dict[int, str] | Sequence[str]] | None = None,
        xname: str = _AttrName.XNAME.value,
        yname: str = _AttrName.YNAME.value,
        zname: str = _AttrName.ZNAME.value,
        idname: str | None = None,  # Well, Polygon, ...
        undef: float | Sequence[float, float] = -999.0,
        xyztype: Literal["well", "points", "polygons"] = "well",
        floatbits: Literal["float32", "float64"] = "float64",
    ):
        logger.info("Running init for: %s", __name__)
        self._df = dataframe

        self._attr_types = {}
        if isinstance(attr_types, dict):
            for name, atype in attr_types.items():
                use_atype = "DISC" if atype.upper() in ("DISC", "INT") else "CONT"
                self._attr_types[name] = _AttrType[use_atype]

        self._attr_records = attr_records if attr_records is not None else {}
        self._xname = xname
        self._yname = yname
        self._zname = zname
        self._idname = idname
        self._floatbits = (
            floatbits if floatbits in ["float32", "float64"] else "float64"
        )

        # undefined data are given by a value, that may be different for cont vs disc
        if isinstance(undef, list):
            self._undef_disc = undef[0]
            self._undef_cont = undef[1]
        else:
            self._undef_disc = undef
            self._undef_cont = undef

        if xyztype == "well":
            self._xyztype = _XYZType.WELL

        self._hash = ("0", "0", "0")

        logger.debug("Initial _attr_types: %s", self._attr_types)
        logger.debug("Initial _attr_records: %s", self._attr_records)
        self.ensure_consistency()
        logger.debug("Initial after consistency chk _attr_types: %s", self._attr_types)
        logger.debug(
            "Initial after consistency chk _attr_records: %s", self._attr_records
        )

    @property
    def dataframe(self):
        return self._df

    data = dataframe  # alias

    @property
    def attr_types(self):
        return self._attr_types

    @property
    def attr_records(self):
        return self._attr_records

    @property
    def xname(self):
        return self._xname

    @xname.setter
    def xname(self, name: str):
        if isinstance(name, str):
            self._xname = name
        else:
            raise ValueError(f"Input name is not a string: {name}")

    @property
    def yname(self):
        return self._yname

    @yname.setter
    def yname(self, name: str):
        if isinstance(name, str):
            self._yname = name
        else:
            raise ValueError(f"Input name is not a string: {name}")

    @property
    def zname(self):
        return self._zname

    @zname.setter
    def zname(self, name: str):
        if isinstance(name, str):
            self._zname = name
        else:
            raise ValueError(f"Input name is not a string: {name}")

    def _infer_attr_dtypes(self):
        """Return as dict on form {"X_UTME": _AttrType.CONT, "FACIES": _AttrType.DISC}.

        There are some important restrictions:
        * The first 3 columns (X Y Z) are always CONT, even if input appears as DISC.
        * A check is made towards existing attr_types; if the key,value pair exists
          already, this function will *not* force a change but keep as is.
        """

        # pandas function that e.g. will convert integer'ish floats to int:
        new_df = self._df.convert_dtypes()

        dlist = new_df.dtypes.to_dict()
        logger.debug("Initial attr_type: %s", self._attr_types)

        datatypes = {}
        for name, dtype in dlist.items():
            if name in self._attr_types:
                # do not change already set attr_types
                datatypes[name] = self._attr_types[name]
                continue

            if name in (self._xname, self._yname, self._zname):
                # force coordinates, first 3 columns, to be CONT
                datatypes[name] = _AttrType.CONT
                continue

            if "float" in str(dtype).lower():
                datatypes[name] = _AttrType.CONT
            elif "int" in str(dtype).lower():
                # although it looks like int, we keep as float since it is not
                # _explicitly_ set, to preserve backward compatibility.
                datatypes[name] = _AttrType.CONT  # CONT being INTENTIONAL!
            else:
                raise RuntimeError(
                    "Log type seems to be something else than float or int for "
                    f"{name}: {dtype}"
                )
        self._attr_types = datatypes
        logger.debug("Processed attr_type: %s", self._attr_types)

    def _ensure_consistency_attr_types(self):
        """Ensure that dataframe and attr_types are consistent.

        attr_types are on form {"GR": "CONT", "ZONES": "DISC", ...}

        The column data in the dataframe takes precedence; i.e. if a column is removed
        in a pandas operation, then attr_types are adapted silently by removing the item
        from the dict.
        """
        # check first if an attr. is removed in dataframe (e.g. by pandas operations)
        logger.debug("Ensure consistency attr_types...")
        for attr_name in list(self._attr_types.keys()):
            if attr_name not in self._df.columns[3:]:
                del self._attr_types[attr_name]

        self._infer_attr_dtypes()

    def _infer_automatic_record(self, attr_name: str):
        """Establish automatic record from name, type and values as first attempt."""
        if self.get_attr_type(attr_name) == _AttrType.CONT.value:
            self._attr_records[attr_name] = CONT_DEFAULT_RECORD
        else:
            # it is a discrete log with missing record; try to find
            # a default one based on current values...
            lvalues = self._df[attr_name].to_numpy().round(decimals=0)
            lvalues = lvalues[~np.isnan(lvalues)]  # remove Nans

            if len(lvalues) > 0:
                lvalues = lvalues.astype("int")
                unique = np.unique(lvalues).tolist()
                codes = {value: str(value) for value in unique}
                if self._undef_disc in codes:
                    del codes[self._undef_disc]
                if UNDEF_DISC in codes:
                    del codes[UNDEF_DISC]
            else:
                codes = None

            self._attr_records[attr_name] = codes

    def _ensure_consistency_attr_records(self):
        """Ensure that data and attr_records are consistent; cf attr_types.

        Important that input attr_types are correct; i.e. run
        _ensure_consistency_attr_types() first!
        """
        for attr_name, dtype in self._attr_types.items():
            logger.debug("attr_name: %s, and dtype: %s", attr_name, dtype)
            if attr_name not in self._attr_records or not isinstance(
                self._attr_records[attr_name],
                (dict, list, tuple),
            ):
                self._infer_automatic_record(attr_name)

            # correct when attr_types is CONT but attr_records for that entry is a dict
            if (
                attr_name in self._attr_records
                and self._attr_types[attr_name] == _AttrType.CONT
                and isinstance(self._attr_records[attr_name], dict)
            ):
                self._attr_records[attr_name] = CONT_DEFAULT_RECORD

    def _ensure_consistency_df_dtypes(self):
        """Ensure that dataframe float32/64 for all logs, except for XYZ -> float64.

        Whether it is float32 or float64 is set by self._floatbits. Float32 will save
        memory but loose some precision. For backward compatibility, float64 is default.
        """

        col = list(self._df)
        logger.debug("columns: %s", col)

        coords_dtypes = [str(entry) for entry in self._df[col[0:3]].dtypes]

        if not all("float64" in entry for entry in coords_dtypes):
            self._df[col[0:3]] = self._df.iloc[:, 0:3].astype("float64")

        attr_dtypes = [str(entry) for entry in self._df[col[3:]].dtypes]

        if not all(self._floatbits in entry for entry in attr_dtypes):
            self._df[col[3:]] = self._df.iloc[:, 3:].astype(self._floatbits)

        for name, attr_type in self._attr_types.items():
            if attr_type == _AttrType.CONT.value:
                logger.debug("Replacing CONT undef...")
                self._df.loc[:, name] = self._df[name].replace(
                    self._undef_cont,
                    np.float64(UNDEF_CONT).astype(self._floatbits),
                )
            else:
                logger.debug("Replacing INT undef...")
                self._df.loc[:, name] = self._df[name].replace(
                    self._undef_disc, np.int32(UNDEF_DISC)
                )
        logger.info("Processed dataframe: %s", list(self._df.dtypes))

    def ensure_consistency(self) -> bool:
        """Ensure that data and attr* are consistent.

        This is important for many operations on the dataframe, an should keep
        attr_types and attr_records 'in sync' with the dataframe.

        * When adding one or columns to the dataframe
        * When removing one or more columns from the dataframe
        * ...

        Returns True is consistency is ran, while False means that no changes have
        occured, hence no consistency checks are done
        """

        # the purpose of this hash check is to avoid spending time on consistency
        # checks if no changes
        hash_proposed = (
            jhash(self._df),
            jhash(self._attr_types),
            jhash(self._attr_records),
        )
        if self._hash == hash_proposed:
            return False

        if list(self._df.columns[:3]) != [self._xname, self._yname, self._zname]:
            raise ValueError(
                f"Dataframe must include '{self._xname}', '{self._yname}' "
                f"and '{self._zname}', got {list(self._df.columns[:3])}"
            )

        # order matters:
        self._ensure_consistency_attr_types()
        self._ensure_consistency_attr_records()
        self._ensure_consistency_df_dtypes()
        self._df.reset_index(drop=True, inplace=True)

        self._hash = (
            jhash(self._df),
            jhash(self._attr_types),
            jhash(self._attr_records),
        )

        return True

    def get_attr_type(self, name: str) -> str:
        """Get the attr_type as string"""
        return self._attr_types[name].name

    def set_attr_type(self, name: str, attrtype: str) -> None:
        """Set a type (DISC, CONT) for a named attribute.

        A bit flexibility is added for attrtype, e.g. allowing "float*" for CONT
        etc, and allow lowercase "cont" for CONT

        """
        logger.debug("Set the attribute type for %s as %s", name, attrtype)
        apply_attrtype = attrtype.upper()

        # allow for optionally using INT and FLOAT in addation to DISC and CONT
        if "FLOAT" in apply_attrtype:
            apply_attrtype = _AttrType.CONT.value
        if "INT" in apply_attrtype:
            apply_attrtype = _AttrType.DISC.value

        if name not in self._attr_types:
            raise ValueError(f"No such log name present: {name}")

        if self.get_attr_type(name) == apply_attrtype:
            logger.debug("Same attr_type as existing, return")
            return

        if apply_attrtype in _AttrType.__members__:
            self._attr_types[name] = _AttrType[apply_attrtype]
        else:
            raise ValueError(
                f"Cannot set wlogtype as {attrtype}, not in "
                f"{list(_AttrType.__members__)}"
            )

        # need to update records with defaults
        self._infer_automatic_record(name)

        self.ensure_consistency()

    def get_attr_record(self, name: str):
        """Get a record for a named attribute."""
        return self._attr_records[name]

    def set_attr_record(self, name: str, record: dict | None) -> None:
        """Set a record for a named log."""

        if name not in self._attr_types:
            raise ValueError(f"No such attr_name: {name}")

        if record is None and self._attr_types[name] == _AttrType.DISC:
            record = {}
        elif record is None and self._attr_types[name] == _AttrType.CONT:
            record = CONT_DEFAULT_RECORD

        if self._attr_types[name] == _AttrType.CONT and isinstance(
            record, (list, tuple)
        ):
            if len(record) == 2:
                self._attr_records[name] = tuple(record)  # prefer as tuple
        elif self._attr_types[name] == _AttrType.CONT and isinstance(record, dict):
            raise ValueError(
                "Cannot set a log record for a continuous log: input record is "
                "dictionary, not a list or tuple"
            )
        elif self._attr_types[name] == _AttrType.DISC and isinstance(record, dict):
            self._attr_records[name] = record
        elif self._attr_types[name] == _AttrType.DISC and not isinstance(record, dict):
            raise ValueError(
                "Input is not a dictionary. Cannot set a log record for a discrete log"
            )
        else:
            raise ValueError(
                "Something went wrong when setting logrecord: "
                f"({self._attr_types[name]} {type(record)})."
            )

        self.ensure_consistency()

    def get_dataframe_copy(
        self,
        infer_dtype: bool = False,
        filled=False,
        fill_value=UNDEF_CONT,
        fill_value_int=UNDEF_DISC,
    ):
        """Get a deep copy of the dataframe, with options.

        If infer_dtype is True, then DISC columns will be of "int32" type, but
        since int32 do not support np.nan, the value for undefined values will be
        ``fill_value_int``
        """
        dfr = self._df.copy(deep=True)
        if infer_dtype:
            for name, attrtype in self._attr_types.items():
                if attrtype.name == _AttrType.DISC.value:
                    dfr[name] = dfr[name].fillna(fill_value_int)
                    dfr[name] = dfr[name].astype("int32")

        if filled:
            dfill = {}
            for attrname in self._df:
                if self._attr_types[attrname] == _AttrType.DISC:
                    dfill[attrname] = fill_value_int
                else:
                    dfill[attrname] = fill_value

            dfr = dfr.fillna(dfill)

        return dfr

    def get_dataframe(self, copy=True):
        """Get the dataframe, as view or deep copy."""
        if copy:
            return self._df.copy(deep=True)

        return self._df

    def set_dataframe(self, dfr: pd.DataFrame):
        """Set the dataframe in a controlled manner, shall be used"""
        # TODO: more checks, and possibly acceptance of lists, dicts?
        if isinstance(dfr, pd.DataFrame):
            self._df = dfr
        else:
            raise ValueError("Input dfr is not a pandas dataframe")
        self.ensure_consistency()

    def rename_attr(self, attrname: str, newname: str):
        """Rename a attribute, e.g. Poro to PORO."""

        if attrname not in list(self._df):
            raise ValueError("Input log does not exist")

        if newname in list(self._df):
            raise ValueError("New log name exists already")

        # rename in dataframe
        self._df.rename(index=str, columns={attrname: newname}, inplace=True)

        self._attr_types[newname] = self._attr_types.pop(attrname)
        self._attr_records[newname] = self._attr_records.pop(attrname)

        self.ensure_consistency()

    def create_attr(
        self,
        attrname: str,
        attr_type: str = Literal[
            _AttrType.CONT.value,  # type: ignore
            _AttrType.DISC.value,  # type: ignore
        ],
        attr_record: dict | None = None,
        value: float = 0.0,
        force: bool = True,
        force_reserved: bool = False,
    ) -> bool:
        """Create a new attribute, e.g. a log."""

        if attrname in list(self._df) and force is False:
            return False

        if attrname in _AttrName.list() and not force_reserved:
            raise ValueError(
                f"The proposed name {attrname} is a reserved name; try another or "
                "set keyword ``force_reserved`` to True ."
                f"Note that the follwoing names are reserved: {_AttrName.list()}"
            )

        self._attr_types[attrname] = _AttrType[attr_type]
        self._attr_records[attrname] = attr_record

        # make a new column
        self._df[attrname] = float(value)
        self.ensure_consistency()
        return True

    def copy_attr(self, attrname: str, new_attrname: str, force: bool = True) -> bool:
        """Copy a attribute to a new name."""

        if new_attrname in list(self._df) and force is False:
            return False

        self._attr_types[new_attrname] = deepcopy(self._attr_types[attrname])
        self._attr_records[new_attrname] = deepcopy(self._attr_records[attrname])

        # make a new column
        self._df[new_attrname] = self._df[attrname].copy()
        self.ensure_consistency()
        return True

    def delete_attr(self, attrname: str | list[str]) -> int:
        """Delete/remove an existing attribute, or list of attributes.

        Returns number of logs deleted
        """
        if not isinstance(attrname, list):
            attrname = [attrname]

        lcount = 0
        for logn in attrname:
            if logn not in list(self._df):
                continue

            lcount += 1
            logger.debug("Actually deleting %s", logn)
            self._df.drop(logn, axis=1, inplace=True)

        self.ensure_consistency()

        return lcount

    def create_relative_hlen(self):
        """Make a relative length of e.g. a well, as a attribute (log)."""
        # extract numpies from XYZ trajectory logs
        xv = self._df[self._xname].values
        yv = self._df[self._yname].values

        distance = []
        previous_x, previous_y = xv[0], yv[0]
        for _, (x, y) in enumerate(zip(xv, yv)):
            distance.append(math.hypot((previous_x - x), (y - previous_y)))
            previous_x, previous_y = x, y

        self._df.loc[:, _AttrName.R_HLEN_NAME.value] = pd.Series(
            np.cumsum(distance), index=self._df.index
        )
        self.ensure_consistency()

    def geometrics(self):
        """Compute geometrical arrays MD, INCL, AZI, as attributes (logs) (~well data).

        These are kind of quasi measurements hence the attributes (logs) will named
        with a Q in front as Q_MDEPTH, Q_INCL, and Q_AZI.

        These attributes will be added to the dataframe.

        TODO: If the mdlogname
        attribute does not exist in advance, it will be set to 'Q_MDEPTH'.

        Returns:
            False if geometrics cannot be computed

        """
        # TODO: rewrite in pure python?
        if self._df.shape[0] < 3:
            raise ValueError(
                f"Cannot compute geometrics. Not enough "
                f"trajectory points (need >3, have: {self._df.shape[0]})"
            )

        # extract numpies from XYZ trajetory logs
        ptr_xv = _get_carray(self._df, self._attr_types, self._xname)
        ptr_yv = _get_carray(self._df, self._attr_types, self._yname)
        ptr_zv = _get_carray(self._df, self._attr_types, self._zname)

        # get number of rows in pandas
        nlen = len(self._df)

        ptr_md = _cxtgeo.new_doublearray(nlen)
        ptr_incl = _cxtgeo.new_doublearray(nlen)
        ptr_az = _cxtgeo.new_doublearray(nlen)

        ier = _cxtgeo.well_geometrics(
            nlen, ptr_xv, ptr_yv, ptr_zv, ptr_md, ptr_incl, ptr_az, 0
        )

        if ier != 0:
            raise XTGeoCLibError(f"XYZ/well_geometrics failed with error code: {ier}")

        dnumpy = _convert_carr_double_np(len(self._df), ptr_md)
        self._df[_AttrName.Q_MD_NAME.value] = pd.Series(dnumpy, index=self._df.index)

        dnumpy = _convert_carr_double_np(len(self._df), ptr_incl)
        self._df[_AttrName.Q_INCL_NAME.value] = pd.Series(dnumpy, index=self._df.index)

        dnumpy = _convert_carr_double_np(len(self._df), ptr_az)
        self._df[_AttrName.Q_AZI_NAME.value] = pd.Series(dnumpy, index=self._df.index)

        # delete tmp pointers
        _cxtgeo.delete_doublearray(ptr_xv)
        _cxtgeo.delete_doublearray(ptr_yv)
        _cxtgeo.delete_doublearray(ptr_zv)
        _cxtgeo.delete_doublearray(ptr_md)
        _cxtgeo.delete_doublearray(ptr_incl)
        _cxtgeo.delete_doublearray(ptr_az)

        self.ensure_consistency()

        return True
