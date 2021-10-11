"""GridProperty (not GridProperies) import functions"""
import xtgeo

from ._gridprop_import_eclrun import (
    decorate_name,
    import_gridprop_from_init_file,
    import_gridprops_from_restart_file,
)
from ._gridprop_import_grdecl import import_bgrdecl_prop, import_grdecl_prop
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

    elif fformat in ["finit", "init"]:
        if grid is None:
            raise ValueError("A grid is required to import init file")
        result = import_gridprop_from_init_file(pfile.file, [name], grid, fracture)
        if len(result) != 1:
            raise ValueError(f"Could not find property {name} in {pfile}")
        result[0]["name"] = decorate_name(result[0]["name"], grid.dualporo, fracture)
        for attr, value in result[0].items():
            setattr(self, "_" + attr, value)
    elif fformat in ["funrst", "unrst"]:
        if grid is None:
            raise ValueError("A grid is required to import restart file")
        if date is None:
            raise ValueError("Restart file, but no date is given")

        if isinstance(date, str):
            if "-" in date:
                date = date.replace("-", "")
        if date not in ("all", "first", "last"):
            dates = [int(date)]
        else:
            dates = date

        result = import_gridprops_from_restart_file(
            pfile.file, [name], dates, grid, fracture, fformat=fformat.lower()
        )
        if len(result) != 1:
            raise ValueError(
                f"Could not find property {name} for {date} in {pfile.file}"
            )
        result[0]["name"] = decorate_name(
            result[0]["name"], grid.dualporo, fracture, result[0]["date"]
        )
        for attr, value in result[0].items():
            setattr(self, "_" + attr, value)

    elif fformat == "grdecl":
        import_grdecl_prop(self, pfile, name=name, grid=grid)

    elif fformat == "bgrdecl":
        import_bgrdecl_prop(self, pfile, name=name, grid=grid)

    elif fformat in ["xtgcpprop", "xtg"]:
        import_xtgcpprop(self, pfile, ijrange=ijrange, zerobased=zerobased)

    else:
        logger.warning("Invalid file format")
        raise ValueError(f"Invalid grid property file format {fformat}")
    return self
