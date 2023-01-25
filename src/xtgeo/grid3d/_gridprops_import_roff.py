from typing import List, Union

from typing_extensions import Literal

import xtgeo
from xtgeo.common.sys import _XTGeoFile

from . import _grid3d_utils as utils
from .grid_property import GridProperty

xtg = xtgeo.common.XTGeoDialog()

logger = xtg.functionlogger(__name__)


def import_roff_gridproperties(
    pfile: _XTGeoFile,
    names: Union[List[str], Literal["all"]],
    strict: bool = True,
) -> List[GridProperty]:
    """Imports list of properties from a roff file.

    Args:
        pfile: Reference to the file
        names: List of names to fetch, can also be "all" to fetch all properties.
        strict: If strict=True, will raise error if key is not found.
    Returns:
        List of GridProperty objects fetched from the ROFF file.
    """
    print(pfile)
    validnames = []

    collectdata = utils.scan_keywords(pfile, fformat="roff")
    for item in collectdata:
        keyname = item[0]
        if keyname.startswith("parameter!name!"):
            # format is 'parameter!name!FIPNUM'
            validnames.append(keyname.split("!").pop())

    usenames = []
    if names == "all":
        usenames = validnames
    else:
        for name in names:
            if name not in validnames:
                if strict:
                    raise ValueError(
                        f"Requested keyword {name} is not in ROFF file, valid "
                        f"entries are {validnames}, set strict=False to warn instead."
                    )
                else:
                    logger.warning(
                        "Requested keyword %s is not in ROFF file. Entry will"
                        "not be read, set strict=True to raise Error instead.",
                        name,
                    )
            else:
                usenames.append(name)

    props = [
        xtgeo.gridproperty_from_file(pfile.file, fformat="roff", name=name)
        for name in usenames
    ]

    return props
