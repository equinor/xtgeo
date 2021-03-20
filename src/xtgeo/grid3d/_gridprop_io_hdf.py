"""GridProperty import / output for HDF format.

Currently there is one gridproperty per HDF file. The name of the gridproperty
is in the _required_ metadata section.

The required metadata are defined in the metdata class and are:



"""
import json
from collections import OrderedDict

import h5py
import hdf5plugin

import xtgeo

xtg = xtgeo.common.XTGeoDialog()

logger = xtg.functionlogger(__name__)

# xgridprop is the GridProperty self instance


def export_hdf5(xgridprop, gfile, compression=None, chunks=False):
    """Export gridprop to h5/hdf5, in prep. and experimental."""
    logger.info("Export to hdf5 xtgeo layout...")

    xgridprop.metadata.required = xgridprop
    meta = xgridprop.metadata.get_metadata()

    if compression and compression == "blosc":
        compression = hdf5plugin.Blosc(
            cname="blosclz", clevel=9, shuffle=hdf5plugin.Blosc.SHUFFLE
        )

    if not chunks:
        chunks = None

    jmeta = json.dumps(meta).encode()

    with h5py.File(gfile.name, "w") as fh5:

        grp = fh5.create_group("CPGridProperty")
        grp.create_dataset(
            "values",
            data=xgridprop.values,
            compression=compression,
            chunks=chunks,
        )

        grp.attrs["metadata"] = jmeta
        grp.attrs["provider"] = "xtgeo"
        grp.attrs["format-idcode"] = 1352 if xgridprop._isdiscrete else 1351

    logger.info("Export to hdf5 xtgeo layout... done!")


def import_hdf5(xgridprop, mfile, ijkrange=None, zerobased=False, name=None):
    """Experimental grid property import using hdf5."""
    #
    reqattrs = xtgeo.MetaDataCPGridProperty.REQUIRED
    ncol2 = nrow2 = nlay2 = 1
    with h5py.File(mfile.name, "r") as h5h:

        grp = h5h["CPGridProperty"]

        idcode = grp.attrs["format-idcode"]
        provider = grp.attrs["provider"]
        if idcode not in (1351, 1352):
            raise ValueError(f"Wrong id code: {idcode}")
        logger.info("Provider is %s", provider)

        jmeta = grp.attrs["metadata"]
        meta = json.loads(jmeta, object_pairs_hook=OrderedDict)

        req = meta["_required_"]

        if name and req["name"] != name:
            raise ValueError(
                f"Requested name {name} != name from metadata: {req['name']}"
            )

        if ijkrange is not None:
            values, ncol2, nrow2, nlay2 = _partial_read(h5h, req, ijkrange, zerobased)
        else:
            values = grp["values"][:, :, :]

    for myattr in reqattrs:
        setattr(xgridprop, "_" + myattr, req[myattr])

    if ijkrange:
        xgridprop._ncol = ncol2
        xgridprop._nrow = nrow2
        xgridprop._nlay = nlay2

    xgridprop._values = values

    xgridprop._metadata.required = xgridprop


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

    dset = h5h["CPGridProperty/values"]
    pv = dset[i1 : i1 + ncol2, j1 : j1 + nrow2, k1 : k1 + nlay2]

    return pv, ncol2, nrow2, nlay2
