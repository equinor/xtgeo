"""GridProperty (not GridProperies) import functions."""
import os
import json
from collections import OrderedDict
import h5py

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
    _roffapiv=1,
    ijrange=None,
    zerobased=False,
):  # _roffapiv for devel.
    """Import grid property from file, and makes an instance of this."""
    # it may be that pfile already is an open file; hence a filehandle
    # instead. Check for this, and skip actions if so
    if not isinstance(pfile, xtgeo._XTGeoFile):
        raise RuntimeError("Internal error, pfile is not a _XTGeoFile instance")

    fformat = _chk_file(self, pfile.name, fformat)

    if fformat == "roff":
        logger.info("Importing ROFF...")
        import_roff(self, pfile, name, grid=grid, _roffapiv=_roffapiv)

    elif fformat.lower() == "init":
        impeclbin(
            self, pfile, name=name, etype=1, date=None, grid=grid, fracture=fracture
        )

    elif fformat.lower() == "unrst":
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

    elif fformat.lower() == "grdecl":
        import_grdecl_prop(self, pfile, name=name, grid=grid)

    elif fformat.lower() == "bgrdecl":
        import_bgrdecl_prop(self, pfile, name=name, grid=grid)

    elif fformat.lower() == "xtgcpprop":
        import_xtgcpprop(self, pfile, ijrange=ijrange, zerobased=zerobased)

    else:
        logger.warning("Invalid file format")
        raise ValueError("Invalid file format")

    # if grid, then append this gridprop to the current grid object

    # ###################################TMP skipped""
    # if grid:
    #     grid.append_prop(self)

    return self


def _chk_file(self, pfile, fformat):

    self._filesrc = pfile

    if os.path.isfile(pfile):
        logger.debug("File %s exists OK", pfile)
    else:
        raise OSError("No such file: {}".format(pfile))

    # work on file extension
    _, fext = os.path.splitext(pfile)
    if fformat is None or fformat == "guess":
        if not fext:
            raise ValueError("File extension missing. STOP")

        fformat = fext.lower().replace(".", "")

    logger.debug("File name to be used is %s", pfile)
    logger.debug("File format is %s", fformat)

    return fformat


def import_hdf5(self, mfile, ijkrange=None, zerobased=False):
    """Experimental grid property import using hdf5."""
    #
    reqattrs = xtgeo.MetaDataCPGridGeometry.REQUIRED
    ncol2 = nrow2 = nlay2 = 1
    with h5py.File(mfile.name, "r") as h5h:

        grp = h5h["GridProperty"]

        idcode = grp.attrs["format-idcode"]
        provider = grp.attrs["provider"]
        if idcode not in (1351, 1352):
            raise ValueError(f"Wrong id code: {idcode}")
        logger.info("Provider is %s", provider)

        jmeta = grp.attrs["metadata"]
        meta = json.loads(jmeta, object_pairs_hook=OrderedDict)

        req = meta["_required_"]

        if ijkrange is not None:
            values, ncol2, nrow2, nlay2 = _partial_read(h5h, req, ijkrange, zerobased)
        else:
            values = grp["values"][:, :, :]

    for myattr in reqattrs:
        setattr(self, "_" + myattr, req[myattr])

    if ijkrange:
        self._ncol = ncol2
        self._nrow = nrow2
        self._nlay = nlay2

    self._values = values

    self._metadata.required = self


def _partial_read(h5h, req, ijkrange, zerobased):
    """Read a partial IJ range."""
    ncol = req["ncol"]
    nrow = req["nrow"]
    nlay = req["nlay"]

    if len(ijkrange) != 6:
        raise ValueError("The ijkrange list must have 6 elements")

    i1, i2, j1, j2, k1, k2 = ijkrange

    if i1 == "min":
        i1 = 0 if zerobased else 1
    if j1 == "min":
        j1 = 0 if zerobased else 1
    if k1 == "min":
        k1 = 0 if zerobased else 1

    if i2 == "max":
        i2 = ncol - 1 if zerobased else ncol
    if j2 == "max":
        j2 = nrow - 1 if zerobased else nrow
    if k2 == "max":
        k2 = nlay - 1 if zerobased else nlay

    if not zerobased:
        i1 -= 1
        i2 -= 1
        j1 -= 1
        j2 -= 1
        k1 -= 1
        k2 -= 1

    ncol2 = i2 - i1 + 1
    nrow2 = j2 - j1 + 1
    nlay2 = k2 - k1 + 1

    if (
        ncol2 < 1
        or ncol2 > ncol
        or nrow2 < 1
        or nrow2 > nrow
        or nlay2 < 1
        or nlay2 > nlay
    ):
        raise ValueError("The ijkrange spesification exceeds boundaries.")

    dset = h5h["GridProperty/values"]
    pv = dset[i1 : i1 + ncol2, j1 : j1 + nrow2, k1 : k1 + nlay2]

    return pv, ncol2, nrow2, nlay2
