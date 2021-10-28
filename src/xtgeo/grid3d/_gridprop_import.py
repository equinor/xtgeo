"""GridProperty (not GridProperies) import functions"""

import xtgeo

from ._gridprop_import_eclrun import import_eclbinary as impeclbin
from ._gridprop_import_grdecl import import_grdecl_prop, import_bgrdecl_prop
from ._gridprop_import_roff import import_roff
from ._gridprop_import_xtgcpprop import import_xtgcpprop

xtg = xtgeo.common.XTGeoDialog()

logger = xtg.functionlogger(__name__)


def from_file(
    self,
    pfile,
    fformat=None,
    name="unknown",
    grid=None,
    date=None,
    fracture=False,
    ijrange=None,
    zerobased=False,
):
    """Import grid property from file, and makes an instance of this."""
    # it may be that pfile already is an open file; hence a filehandle
    # instead. Check for this, and skip actions if so
    if not isinstance(pfile, xtgeo._XTGeoFile):
        raise RuntimeError("Internal error, pfile is not a _XTGeoFile instance")

    pfile.check_file(raiseerror=OSError)

    if fformat is None or fformat == "guess":
        fformat = pfile.detect_fformat(suffixonly=True)
    fformat = fformat.lower()

    if fformat in ["roff", "roff_binary", "roff_ascii"]:
        logger.info("Importing ROFF...")
        import_roff(self, pfile, name)

    elif fformat == "init":
        impeclbin(
            self, pfile, name=name, etype=1, date=None, grid=grid, fracture=fracture
        )

    elif fformat == "unrst":
        if date is None:
            raise ValueError("Restart file, but no date is given")

        if isinstance(date, str):
            if "-" in date:
                date = int(date.replace("-", ""))
            elif date == "first":
                date = 0
            elif date == "last":
                date = 9
            else:
                date = int(date)

        if not isinstance(date, int):
            raise RuntimeError("Date is not int format")

        impeclbin(
            self, pfile, name=name, etype=5, date=date, grid=grid, fracture=fracture
        )

    elif fformat == "grdecl":
        import_grdecl_prop(self, pfile, name=name, grid=grid)

    elif fformat == "bgrdecl":
        import_bgrdecl_prop(self, pfile, name=name, grid=grid)

    elif fformat in ["xtg", "xtgcpprop"]:
        import_xtgcpprop(self, pfile, ijrange=ijrange, zerobased=zerobased)

    else:
        logger.warning("Invalid file format")
        raise ValueError(f"Invalid file format {fformat}")

    return self
