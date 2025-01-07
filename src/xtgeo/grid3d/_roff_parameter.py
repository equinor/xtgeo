from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import roffio

from xtgeo.common.constants import UNDEF_INT_LIMIT, UNDEF_LIMIT

if TYPE_CHECKING:
    from xtgeo.common.types import FileLike

    from .grid_property import GridProperty


@dataclass
class RoffParameter:
    """
    Roff parameter contains a parameter (1 value per grid cell) in a roff file.

    Parameters are either discrete (int) or floating point. Discrete values can
    come with code names giving the meaning of the values. The list code_names
    gives the name for each value in code_values.

    The value -999 for discrete parameters (255 for byte valued discrete
    parameters) and -999.0 for floating point parameters are used undefined
    value.

    Args:
        nx (int): The number of cells in x direction.
        ny (int): The number of cells in y direction.
        nz (int): The number of cells in z direction.
        names (str): The name of the parameter
        values (array of int32 or float, or bytes): One value per cell in c
            order.
        code_names (List of str): The name of the coded value.
        code_values (array of int32): The code values.
    """

    nx: int
    ny: int
    nz: int

    name: str
    values: np.ndarray | bytes

    code_names: list[str] | None = None
    code_values: np.ndarray | None = None

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, RoffParameter):
            return False
        return (
            self.nx == other.nx
            and self.ny == other.ny
            and self.nz == other.nz
            and self.name == other.name
            and np.array_equal(self.values, other.values)
            and self.same_codes(other)
        )

    def same_codes(self, other: RoffParameter) -> bool:
        """
        Args:
            other (RoffParameter): Any roff parameter
        Returns:
            True if the roff parameters have the same coded values.
        """
        if self.code_names is None:
            return other.code_names is None
        if self.code_values is None:
            return other.code_values is None

        if other.code_names is None:
            return self.code_names is None
        if other.code_values is None:
            return self.code_values is None

        return dict(zip(self.code_values, self.code_names)) == dict(
            zip(other.code_values, other.code_names)
        )

    @property
    def undefined_value(self) -> int | float:
        """
        Returns:
            The undefined value for the type of values in the
            roff parameter (either 255, -999, or -999.0)
        """
        if isinstance(self.values, bytes) or np.issubdtype(self.values.dtype, np.uint8):
            return 255
        if np.issubdtype(self.values.dtype, np.integer):
            return -999
        if np.issubdtype(self.values.dtype, np.floating):
            return -999.0
        raise ValueError(f"Parameter values of unsupported type {type(self.values)}")

    @property
    def is_discrete(self) -> bool:
        """
        Returns:
            True if the RoffParameter is a discrete type
        """
        return bool(
            isinstance(self.values, bytes)
            or np.issubdtype(self.values.dtype, np.integer)
        )

    def xtgeo_codes(self) -> dict[int, str]:
        """
        Returns:
            The discrete codes of the parameter in the format of
            xtgeo.GridProperty.
        """
        return (
            dict(zip(self.code_values, self.code_names))
            if self.code_names is not None and self.code_values is not None
            else {}
        )

    def xtgeo_values(self) -> np.ndarray:
        """
        Args:
            The value to use for undefined. Defaults to that defined by
            roff.
        Returns:
            The values in the format of xtgeo grid property
        """
        if isinstance(self.values, bytes):
            vals: np.ndarray = np.ndarray(len(self.values), np.uint8, self.values)
        else:
            vals = self.values.copy()

        vals = np.flip(vals.reshape((self.nx, self.ny, self.nz)), -1)
        vals = vals.astype(np.int32) if self.is_discrete else vals.astype(np.float64)
        return np.ma.masked_values(vals, self.undefined_value)

    @staticmethod
    def from_xtgeo_grid_property(gridproperty: GridProperty) -> RoffParameter:
        """
        Args:
            xtgeo_grid_property (xtgeo.GridProperty): Any xtgeo.GridProperty
        Returns:
            That grid property as a RoffParameter
        """
        code_names = None
        code_values = None
        if gridproperty.isdiscrete:
            code_names = list(gridproperty.codes.values())
            code_values = np.array(list(gridproperty.codes.keys()), dtype=np.int32)

        if not np.ma.isMaskedArray(gridproperty.values):
            values = np.ma.masked_greater(
                gridproperty.values,
                UNDEF_INT_LIMIT if gridproperty.isdiscrete else UNDEF_LIMIT,
            )
        else:
            values = gridproperty.values

        if gridproperty.isdiscrete:
            values = values.astype(np.int32).filled(-999)
        else:
            # Although the roff format can contain double,
            # double typed parameters are not read by RMS so we
            # need to convert to float32 here
            values = values.astype(np.float32).filled(-999.0)

        return RoffParameter(
            nx=gridproperty.ncol,
            ny=gridproperty.nrow,
            nz=gridproperty.nlay,
            name=gridproperty.name or "",
            values=np.asarray(np.flip(values, -1).ravel()),
            code_names=code_names,
            code_values=code_values,
        )

    def to_file(
        self,
        filelike: FileLike,
        roff_format: roffio.Format = roffio.Format.BINARY,
    ) -> None:
        """
        Writes the RoffParameter to a roff file
        Args:
            filelike (str or byte stream): The file to write to.
            roff_format (roffio.Format): The format to write the file in.
        """
        data: dict[str, dict[str, Any]] = {
            "filedata": {"filetype": "parameter"},
            "dimensions": {"nX": self.nx, "nY": self.ny, "nZ": self.nz},
            "parameter": {"name": self.name},
        }
        if self.code_names is not None:
            data["parameter"]["codeNames"] = list(self.code_names)
        if self.code_values is not None:
            data["parameter"]["codeValues"] = self.code_values
        data["parameter"]["data"] = self.values
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", r"casting array")
            roffio.write(filelike, data, roff_format=roff_format)

    @staticmethod
    def from_file(filelike: FileLike, name: str | None = None) -> RoffParameter:
        """
        Read a RoffParameter from a roff file
        Args:
            filelike (str or byte stream): The file to read from.
            name(str or None): The name of the parameter to get from the file,
                if name=None, returns the first parameter read.
        Returns:
            The RoffGrid in the roff file.
        """

        def should_skip_parameter(tag: str, key: str) -> bool:
            if tag == "parameter" and key[0] == "name":
                return not (name is None or key[1] == name)
            return False

        translate_kws = {
            "dimensions": {"nX": "nx", "nY": "ny", "nZ": "nz"},
            "parameter": {
                "name": "name",
                "data": "values",
                "codeValues": "code_values",
                "codeNames": "code_names",
            },
        }
        optional_keywords = defaultdict(
            list,
            {"parameter": ["codeValues", "codeNames"]},
        )
        # The found dictionary contains all tags/tagkeys which we are
        # interested in with None as the initial value. We go through the
        # tag/tagkeys in the file and replace as they are found.
        found = {
            tag_name: {key_name: None for key_name in tag_keys}
            for tag_name, tag_keys in translate_kws.items()
        }
        found["filedata"] = {"filetype": None}
        with roffio.lazy_read(filelike) as tag_generator:
            for tag, keys in tag_generator:
                if tag in found:
                    if tag == "parameter" and found["parameter"]["data"] is not None:
                        # We have already found the right parameter so skip
                        # reading and potentially overwriting
                        continue
                    # We do not destruct keys yet as this fetches the value too early.
                    # key is not a tuple but an object that fetches the value when
                    # __getitem__ is called.
                    for key in keys:
                        if should_skip_parameter(tag, key):
                            # Found a parameter, but not the one we are looking for
                            # reset and look on
                            for key_name in found["parameter"]:
                                found["parameter"][key_name] = None
                            break
                        if key[0] in found[tag]:
                            if found[tag][key[0]] is not None:
                                raise ValueError(
                                    f"Multiple tag, tagkey pair {tag}, {key[0]}"
                                    f" in {filelike}"
                                )
                            found[tag][key[0]] = key[1]

        if name is not None and (found["parameter"]["name"] != name):
            raise ValueError(
                "Did not find parameter"
                f" {name} in roff file, got {found['parameter']['name']}"
            )

        for tag_name, keys in found.items():
            for key_name, value in keys.items():
                if value is None and key_name not in optional_keywords[tag_name]:
                    raise ValueError(
                        f"Missing non-optional keyword {tag_name}:{key_name}"
                    )

        filetype = found["filedata"]["filetype"]
        if filetype not in ["grid", "parameter"]:
            raise ValueError(
                f"File {filelike} did not"
                f" have filetype parameter or grid, found {filetype}"
            )

        roff: dict[str, Any] = {}
        for tag, tag_keys in translate_kws.items():
            for key, translated in tag_keys.items():
                if found[tag][key] is not None:
                    roff[translated] = found[tag][key]

        return RoffParameter(
            nx=roff["nx"],
            ny=roff["ny"],
            nz=roff["nz"],
            name=roff["name"],
            values=roff["values"],
            code_names=roff.get("code_names"),
            code_values=roff.get("code_values"),
        )
