from __future__ import annotations

import io
import json
import struct
from contextlib import ExitStack
from copy import deepcopy
from typing import IO, TYPE_CHECKING, Any, Literal

import h5py
import hdf5plugin
import roffio

from xtgeo.common import null_logger

from ._egrid import EGrid
from ._grdecl_grid import GrdeclGrid
from ._roff_grid import RoffGrid

logger = null_logger(__name__)

if TYPE_CHECKING:
    import numpy as np

    from xtgeo.common.types import FileLike
    from xtgeo.io._file import FileWrapper
    from xtgeo.metadata.metata import MetaData

    from .grid import Grid

# valid byte settings for xtgeo/hdf5 based export e.g. 441 means
# 4byte float for coord, 4byte float for zcorn, and 1 byte Int for actnums.
# Native XTGeo is 844
VALIDXTGFMT = (221, 421, 441, 444, 841, 844, 881, 884)


def export_roff(
    grid: Grid, gfile: FileLike, roff_format: Literal["ascii", "binary"] = "binary"
) -> None:
    """Export grid to ROFF format."""
    if roff_format == "binary":
        RoffGrid.from_xtgeo_grid(grid).to_file(gfile, roff_format=roffio.Format.BINARY)
    elif roff_format == "ascii":
        RoffGrid.from_xtgeo_grid(grid).to_file(gfile, roff_format=roffio.Format.ASCII)
    else:
        raise ValueError(
            "Incorrect format specifier in export_roff,"
            f" expected 'binary' or 'ascii, got {roff_format}"
        )


def export_grdecl(grid: Grid, gfile: FileLike, mode: int) -> None:
    """Export grid to Eclipse GRDECL format (ascii, mode=1) or binary (mode=0)."""
    fileformat = "grdecl" if mode == 1 else "bgrdecl"
    GrdeclGrid.from_xtgeo_grid(grid).to_file(gfile, fileformat=fileformat)


def export_egrid(grid: Grid, gfile: FileLike) -> None:
    """Export grid to Eclipse EGRID format, binary."""
    EGrid.from_xtgeo_grid(grid).to_file(gfile, fileformat="egrid")


def export_fegrid(grid: Grid, gfile: FileLike) -> None:
    """Export grid to Eclipse FEGRID format, ascii."""
    EGrid.from_xtgeo_grid(grid).to_file(gfile, fileformat="fegrid")


def export_xtgcpgeom(grid: Grid, gfile: FileWrapper, subformat: int = 844) -> None:
    """Export grid to binary XTGeo xtgcpgeom format, in prep. and experimental."""
    logger.info("Export to native binary xtgeo...")

    grid._xtgformat2()
    logger.info("Export to native binary xtgeo...(2)")

    grid.metadata.required = grid
    meta = grid.metadata.get_metadata()

    # subformat processing, indicating number of bytes per datatype
    # here, 844 is native XTGeo (float64, float32, int32)
    dtypecoo, dtypezco, dtypeact, meta = _define_dtypes(grid, subformat, meta)

    coordsv, zcornsv, actnumsv = _transform_vectors(
        grid, meta, dtypecoo, dtypezco, dtypeact
    )

    prevalues = (1, 1301, int(subformat), grid.ncol, grid.nrow, grid.nlay)
    mystruct = struct.Struct("= i i i q q q")
    hdr = mystruct.pack(*prevalues)

    # Convert StringIO to BytesIO as this is a binary format
    with ExitStack() as stack:
        if isinstance(gfile.file, io.StringIO):
            data = gfile.file.getvalue().encode("utf-8")
            fout: IO[Any] = io.BytesIO(data)
        elif isinstance(gfile.file, io.BytesIO):
            fout = gfile.file
        else:
            fout = stack.enter_context(open(gfile.file, "wb"))
        fout.write(hdr)
        coordsv.tofile(fout)
        zcornsv.tofile(fout)
        actnumsv.tofile(fout)
        fout.write("\nXTGMETA.v01\n".encode())
        fout.write(json.dumps(meta).encode())

    logger.info("Export to native binary xtgeo... done!")


def export_hdf5_cpgeom(
    grid: Grid,
    gfile: FileWrapper,
    compression: Literal["blosc"] | None = None,
    chunks: bool = False,
    subformat: int = 844,
) -> None:
    """Export grid to h5/hdf5, in prep. and experimental."""
    grid._xtgformat2()

    logger.debug("Export to hdf5 xtgeo layout...")

    grid.metadata.required = grid
    meta = grid.metadata.get_metadata()

    if compression == "blosc":
        compression = hdf5plugin.Blosc(
            cname="blosclz", clevel=9, shuffle=hdf5plugin.Blosc.SHUFFLE
        )

    dtypecoo, dtypezco, dtypeact, meta = _define_dtypes(grid, subformat, meta)
    coordsv, zcornsv, actnumsv = _transform_vectors(
        grid, meta, dtypecoo, dtypezco, dtypeact
    )
    jmeta = json.dumps(meta).encode()

    with h5py.File(gfile.name, "w") as fh5:
        grp = fh5.create_group("CornerPointGeometry")
        grp.create_dataset(
            "coord",
            data=coordsv,
            compression=compression,
            chunks=chunks or None,
        )
        grp.create_dataset(
            "zcorn",
            data=zcornsv,
            compression=compression,
            chunks=chunks or None,
        )
        grp.create_dataset(
            "actnum",
            data=actnumsv,
            compression=compression,
            chunks=chunks or None,
        )

        grp.attrs["metadata"] = jmeta
        grp.attrs["provider"] = "xtgeo"
        grp.attrs["format-idcode"] = 1301

    logger.debug("Export to hdf5 xtgeo layout... done!")


def _define_dtypes(grid: Grid, fmt: int, meta: MetaData) -> tuple[str, str, str, str]:
    # subformat processing, indicating number of bytes per datatype
    # here, 844 is native XTGeo (float64, float32, int32)
    logger.debug("Define dtypes...")

    newmeta = deepcopy(meta)

    if int(fmt) not in VALIDXTGFMT:
        raise ValueError(f"The subformat value is not valid, must be in: {VALIDXTGFMT}")

    if fmt <= 444:
        # the float() is for JSON serialization which cannot handle float32
        newmeta["_required_"]["xshift"] = float(grid._coordsv[:, :, 0::3].mean())
        newmeta["_required_"]["yshift"] = float(grid._coordsv[:, :, 1::3].mean())
        newmeta["_required_"]["zshift"] = float(grid._zcornsv.mean())

    nbytecoord, nbytezcorn, nbyteactnum = [int(nbyte) for nbyte in str(fmt)]

    dtype_coord = "float" + str(nbytecoord * 8)
    dtype_zcorn = "float" + str(nbytezcorn * 8)
    dtype_actnum = "int" + str(nbyteactnum * 8)

    logger.debug("Define dtypes... done!")
    return dtype_coord, dtype_zcorn, dtype_actnum, newmeta


def _transform_vectors(
    grid: Grid, meta: MetaData, dtypecoo: str, dtypezco: str, dtypeact: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logger.debug("Transform vectors...")

    xshift = meta["_required_"]["xshift"]
    yshift = meta["_required_"]["yshift"]
    zshift = meta["_required_"]["zshift"]

    if xshift != 0.0 or yshift != 0.0 or zshift != 0.0:
        coordsv = grid._coordsv.copy()
        zcornsv = grid._zcornsv.copy()
        actnumsv = grid._actnumsv.copy()
        coordsv[:, :, 0::3] -= xshift
        coordsv[:, :, 1::3] -= yshift
        coordsv[:, :, 2::3] -= zshift
        zcornsv -= zshift
    else:
        coordsv = grid._coordsv
        zcornsv = grid._zcornsv
        actnumsv = grid._actnumsv

    if dtypecoo != "float64":
        coordsv = coordsv.astype(dtypecoo)

    if dtypecoo != "float32":
        zcornsv = zcornsv.astype(dtypezco)

    if dtypecoo != "int32":
        actnumsv = actnumsv.astype(dtypeact)

    logger.debug("Transform vectors... done!")
    return coordsv, zcornsv, actnumsv
