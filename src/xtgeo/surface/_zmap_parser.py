"""ZMAP plus parsing.

cf https://saurabhkukade.com/posts/2020/07/understanding-zmap-file-format/

Note also from example here:
https://raw.githubusercontent.com/abduhbm/zmapio/main/examples/NSLCU.dat
that header lines may end with trailing comma!
"""

import dataclasses
import inspect
from functools import wraps
from pathlib import Path
from typing import Optional

import numpy as np


@dataclasses.dataclass
class ZMAPSurface:
    nrow: int
    ncol: int
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    node_width: int
    precision: int
    start_column: int
    nan_value: float
    nr_nodes_per_line: int
    values: Optional[np.array] = None

    def __post_init__(self):
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if field.type in (int, float) and not isinstance(value, field.type):
                setattr(self, field.name, field.type(value))


def takes_stream(name, mode):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            kwargs = inspect.getcallargs(func, *args, **kwargs)
            if name in kwargs and isinstance(kwargs[name], (str, Path)):
                with open(kwargs[name], mode) as f:
                    kwargs[name] = f
                    return func(**kwargs)
            else:
                return func(**kwargs)

        return wrapper

    return decorator


def parse_header(zmap_data):
    keys = {}
    line_nr = 0
    for line in zmap_data:
        if is_comment(line):
            continue
        try:
            line = [entry.strip() for entry in line.split(",")]
            if not line[-1]:
                line.pop()  # deal with input lines ending with comma ','
            if line_nr == 0:
                _, identifier, keys["nr_nodes_per_line"] = line
                if identifier != "GRID":
                    raise ZMAPFormatException(
                        f"Expected GRID as second entry in line, "
                        f"got: {identifier} in line: {line}"
                    )
            elif line_nr == 1:
                (
                    keys["node_width"],
                    dft_nan_value,
                    user_nan_value,
                    keys["precision"],
                    keys["start_column"],
                ) = line
                keys["nan_value"] = (
                    user_nan_value if not dft_nan_value else dft_nan_value
                )
            elif line_nr == 2:
                (
                    keys["nrow"],
                    keys["ncol"],
                    keys["xmin"],
                    keys["xmax"],
                    keys["ymin"],
                    keys["ymax"],
                ) = line
            elif line_nr == 3:
                _, _, _ = line
            elif line_nr >= 4 and line[0] != "@":
                raise ZMAPFormatException(
                    f"Did not reach the values section, expected @, found: {line}"
                )
            else:
                return keys
        except ValueError as err:
            raise ZMAPFormatException(f"Failed to unpack line: {line}") from err
        line_nr += 1
    raise ZMAPFormatException("End reached without complete header")


def is_comment(line):
    if line.startswith("!") or line.startswith("+"):
        return True
    return False


def parse_values(zmap_data, nan_value):
    """Parse actual values in zmap plus ascii files.

    Note that header's node_width and nr_nodes_per_line in ZMAP header are not applied,
    meaning that values import here is more tolerant than original zmap spec.
    """
    values = []
    for line in zmap_data:
        if is_comment(line):
            continue
        else:
            values += line.split()
    values = np.array(values, dtype=np.float32)
    values = np.ma.masked_equal(values, nan_value)
    return values


@takes_stream("zmap_file", "r")
def parse_zmap(zmap_file, load_values=True):
    header = parse_header(zmap_file)
    zmap_data = ZMAPSurface(**header)
    if load_values:
        zmap_data.values = parse_values(zmap_file, zmap_data.nan_value)
    return zmap_data


class ZMAPFormatException(Exception):
    pass
