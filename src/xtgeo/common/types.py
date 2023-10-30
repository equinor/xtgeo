"""
This file contains commen types used in xtgeo, keep it free some logic.
"""

from io import BytesIO, StringIO
from pathlib import Path
from typing import Union

FileLike = Union[str, Path, StringIO, BytesIO]
