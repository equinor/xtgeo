import warnings
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import roffio

from xtgeo.common.constants import UNDEF_INT_LIMIT, UNDEF_LIMIT


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
    values: Union[np.ndarray, bytes]

    code_names: Optional[List[str]] = None
    code_values: Optional[np.ndarray] = None

    def __eq__(self, other):
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

    def same_codes(self, other):
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
    def undefined_value(self):
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

    @property
    def is_discrete(self):
        """
        Returns:
            True if the RoffParameter is a discrete type
        """
        return isinstance(self.values, bytes) or np.issubdtype(
            self.values.dtype, np.integer
        )

    def xtgeo_codes(self):
        """
        Returns:
            The discrete codes of the parameter in the format of
            xtgeo.GridProperty.
        """
        if self.code_names is not None and self.code_values is not None:
            return dict(zip(self.code_values, self.code_names))
        else:
            return dict()

    def xtgeo_values(self):
        """
        Args:
            The value to use for undefined. Defaults to that defined by
            roff.
        Returns:
            The values in the format of xtgeo grid property
        """
        vals = self.values
        if isinstance(vals, bytes):
            vals = np.ndarray(len(vals), np.uint8, vals)
        vals = vals.copy()
        vals = np.flip(vals.reshape((self.nx, self.ny, self.nz)), -1)

        if self.is_discrete:
            vals = vals.astype(np.int32)
        else:
            vals = vals.astype(np.float64)

        return np.ma.masked_values(vals, self.undefined_value)

    @staticmethod
    def from_xtgeo_grid_property(xtgeo_grid_property):
        """
        Args:
            xtgeo_grid_property (xtgeo.GridProperty): Any xtgeo.GridProperty
        Returns:
            That grid property as a RoffParameter
        """
        code_names = None
        code_values = None
        if xtgeo_grid_property.isdiscrete:
            code_names = list(xtgeo_grid_property.codes.values())
            code_values = np.array(
                list(xtgeo_grid_property.codes.keys()), dtype=np.int32
            )

        values = xtgeo_grid_property.values
        if not np.ma.isMaskedArray(values):
            if xtgeo_grid_property.isdiscrete:
                values = np.ma.masked_greater(values, UNDEF_INT_LIMIT)
            else:
                values = np.ma.masked_greater(values, UNDEF_LIMIT)

        if xtgeo_grid_property.isdiscrete:
            values = values.astype(np.int32).filled(-999)
        else:
            # Although the roff format can contain double,
            # double typed parameters are not read by RMS so we
            # need to convert to float32 here
            values = values.astype(np.float32).filled(-999.0)

        return RoffParameter(
            *xtgeo_grid_property.dimensions,
            name=xtgeo_grid_property.name,
            values=np.asarray(np.flip(values, -1).ravel()),
            code_names=code_names,
            code_values=code_values,
        )

    def to_file(self, filelike, roff_format=roffio.Format.BINARY):
        """
        Writes the RoffParameter to a roff file
        Args:
            filelike (str or byte stream): The file to write to.
            roff_format (roffio.Format): The format to write the file in.
        """
        data = OrderedDict(
            {
                "filedata": {"filetype": "parameter"},
                "dimensions": {"nX": self.nx, "nY": self.ny, "nZ": self.nz},
                "parameter": {"name": self.name},
            }
        )
        if self.code_names is not None:
            data["parameter"]["codeNames"] = list(self.code_names)
        if self.code_values is not None:
            data["parameter"]["codeValues"] = self.code_values
        data["parameter"]["data"] = self.values
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", r"casting array")
            roffio.write(filelike, data, roff_format=roff_format)

    @staticmethod
    def from_file(filelike, name=None):
        """
        Read a RoffParameter from a roff file
        Args:
            filelike (str or byte stream): The file to read from.
            name(str or None): The name of the parameter to get from the file,
                if name=None, returns the first parameter read.
        Returns:
            The RoffGrid in the roff file.
        """

        def should_skip_parameter(tag, key):
            if tag == "parameter" and key[0] == "name":
                if name is None or key[1] == name:
                    return False
                return True
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
            tag_name: {key_name: None for key_name in tag_keys.keys()}
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
                            for key_name in found["parameter"].keys():
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

        return RoffParameter(
            **{
                translated: found[tag][key]
                for tag, tag_keys in translate_kws.items()
                for key, translated in tag_keys.items()
                if found[tag][key] is not None
            }
        )
