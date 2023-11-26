from __future__ import annotations

import io
from typing import TYPE_CHECKING, Literal

import roffio

import xtgeo
from xtgeo.common import null_logger

if TYPE_CHECKING:
    from xtgeo.common.sys import _XTGeoFile

    from .grid_property import GridProperty

logger = null_logger(__name__)


def import_roff_gridproperties(
    pfile: _XTGeoFile,
    names: list[str] | Literal["all"],
    strict: bool = True,
) -> list[GridProperty]:
    """
    Imports a list of properties from a ROFF file.

    Parameters:
        pfile:
            Reference to the file.
        names:
            List of names to fetch, can also be "all" to fetch all properties.
        strict:
            If strict=True, will raise error if key is not found.
            Defaults to True.

    Returns:
        List of GridProperty objects fetched from the ROFF file.

    """
    validnames = set()
    with roffio.lazy_read(pfile.file) as contents:
        for tagname, tagkeys in contents:
            for keyname, values in tagkeys:
                if tagname == "parameter" and keyname == "name":
                    validnames.add(values)

    # Rewind if this file is in memory
    if isinstance(pfile.file, (io.BytesIO, io.StringIO)):
        pfile.file.seek(0)

    usenames = set()
    if names == "all":
        usenames = validnames
    else:
        for name in names:
            if name in validnames:
                usenames.add(name)
                continue

            if strict:
                raise ValueError(
                    f"Requested keyword {name} is not in ROFF file, valid "
                    f"entries are {validnames}, set strict=False to warn instead."
                )
            logger.warning(
                "Requested keyword %s is not in ROFF file. Entry will"
                "not be read, set strict=True to raise Error instead.",
                name,
            )

    props = [
        xtgeo.gridproperty_from_file(pfile.file, fformat="roff", name=name)
        for name in usenames
    ]
    return props
