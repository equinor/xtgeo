"""
This file contains commen types used in xtgeo, keep it free some logic.
"""

from io import BytesIO, StringIO
from pathlib import Path
from typing import NamedTuple, Union


class Dimensions(NamedTuple):
    """Class for a Dimensions NamedTuple"""

    ncol: int
    nrow: int
    nlay: int


FileLike = Union[str, Path, StringIO, BytesIO]
