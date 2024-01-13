from __future__ import annotations

import io
from typing import TYPE_CHECKING, Generator, Literal

import roffio

import xtgeo
from xtgeo.common import null_logger

if TYPE_CHECKING:
    from xtgeo.io._file import FileWrapper

    from .grid_property import GridProperty

logger = null_logger(__name__)


def read_roff_properties(xtg_file: FileWrapper) -> Generator[str, None, None]:
    """Generates parameter names from roff files.

    Args:
        xtg_file: The FileWrapper representing a roff file.

    Returns:
        Names of parameters within the roff file.
    """
    is_stream = isinstance(xtg_file.file, (io.BytesIO, io.StringIO))
    try:
        if is_stream:
            mark = xtg_file.file.tell()
        with roffio.lazy_read(xtg_file.file) as roff_iter:
            for tag_name, tag_group in roff_iter:
                for keyword, value in tag_group:
                    if tag_name == "parameter" and keyword == "name":
                        yield value
    finally:
        if is_stream:
            xtg_file.file.seek(mark)


def import_roff_gridproperties(
    pfile: FileWrapper,
    names: list[str] | Literal["all"] = "all",
    strict: bool = True,
) -> list[GridProperty]:
    """
    Imports a list of properties from a ROFF file.

    Args:
        pfile: Reference to the file.
        names: List of names to fetch, can also be "all" to fetch all properties.
        strict: If strict=True, will raise error if key is not found.
            Defaults to True.

    Returns:
        List of GridProperty objects fetched from the ROFF file.
    """
    if names == "all":
        return [
            xtgeo.gridproperty_from_file(pfile.file, fformat="roff", name=name)
            for name in read_roff_properties(pfile)
        ]

    prop_names = list(read_roff_properties(pfile))
    props = []
    for name in names:
        if name in prop_names:
            props.append(
                xtgeo.gridproperty_from_file(pfile.file, fformat="roff", name=name)
            )
        elif strict:
            raise ValueError(
                f"Requested keyword {name} is not in ROFF file. Valid "
                f"entries are {prop_names}. Set strict=False to warn instead."
            )
        else:
            logger.warning(
                "Requested keyword %s is not in ROFF file. Entry will"
                "not be read. Set strict=True to raise Error instead.",
                name,
            )
    return props
