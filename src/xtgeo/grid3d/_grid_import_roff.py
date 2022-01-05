# coding: utf-8
"""Private module, Grid Import private functions for ROFF format."""

import pathlib
import tempfile
import warnings
from contextlib import contextmanager

import xtgeo

from ._roff_grid import RoffGrid

xtg = xtgeo.common.XTGeoDialog()

logger = xtg.functionlogger(__name__)


def match_xtgeo_214_header(header: bytes) -> bool:
    """
    Check whether the start of a binary file matches
    the problematic xtgeo version 2.14 contents.

    Params:
      header: the start of a file

    """
    first_part_match = header.startswith(
        b"roff-bin\0"
        b"#ROFF file#\0"
        b"#Creator: CXTGeo subsystem of XTGeo by JCR#\0"
        b"tag\0filedata\0"
        b"int\0byteswaptest\0"
    )
    problem_area_match = header[99:].startswith(
        b"char\0filetype\0grid\0char\0creationDate\0UNKNOWNendtag"
    )
    return first_part_match and problem_area_match


def replace_xtgeo_214_header(header):
    """
    Given that match_xtgeo_214_header(header), inserts
    a \0 in the correct place.
    """
    return header[:143] + b"\0" + header[143:]


@contextmanager
def handle_deprecated_xtgeo_roff_file(filelike):
    """
    A contextmanager that inplace fixes grid roff files
    that were written by XTGeo prior to version 2.15. This
    backwards compatibility should eventually be deprecated

    Example::

       with handle_deprecated_xtgeo_roff_file("grid.roff") as converted_grid:
          roff_grid = RoffGrid.from_file(converted_file)


    Before version 2.15, grids with _xtgformat=1 would be written with missing
    '\\0' after the creationDate. However, it would also silently read that
    file without any issues in the final product. roffio is less leanient when
    it comes to the format it will accept and so does not recover. Luckily, the
    creationDate was set to 'UNKNOWN' so we can be fairly certain we replace
    correctly.

    """
    header = None
    inhandle = filelike
    close = False
    if isinstance(filelike, (str, pathlib.Path)):
        inhandle = open(filelike, "rb")
        close = True
    goback = inhandle.tell()
    header = inhandle.read(200)

    if match_xtgeo_214_header(header):
        name = "buffer"
        if hasattr(filelike, "name"):
            name = filelike.name
        warnings.warn(
            f"The roff file {name} contains nonstandard but harmless roff"
            " format detail written by XTGeo version <=2.14. Reading of such files"
            " is deprecated, consider re-exporting the file with XTGeo version >=2.15.3"
        )
        new_header = replace_xtgeo_214_header(header)

        with tempfile.NamedTemporaryFile() as outhandle:
            outhandle.write(new_header)
            inhandle.seek(200)
            outhandle.write(inhandle.read())
            outhandle.seek(0)
            yield outhandle
            if close:
                inhandle.close()
            else:
                inhandle.seek(goback)

    else:
        if close:
            inhandle.close()
        else:
            inhandle.seek(goback)

        yield filelike


def import_roff(gfile):
    with handle_deprecated_xtgeo_roff_file(gfile._file) as converted_file:
        roff_grid = RoffGrid.from_file(converted_file)
    return {
        "actnumsv": roff_grid.xtgeo_actnum(),
        "coordsv": roff_grid.xtgeo_coord(),
        "zcornsv": roff_grid.xtgeo_zcorn(),
        "subgrids": roff_grid.xtgeo_subgrids(),
    }
