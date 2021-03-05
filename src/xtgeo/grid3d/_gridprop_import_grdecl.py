"""Importing grid props from GRDECL, ascii or binary"""


from contextlib import contextmanager

import numpy as np
import numpy.ma as ma

import xtgeo

from . import _grid3d_utils as utils
from . import _grid_eclbin_record as _eclbin

xtg = xtgeo.common.XTGeoDialog()

logger = xtg.functionlogger(__name__)


def import_bgrdecl_prop(self, pfile, name="unknown", grid=None):
    """Import prop for binary files with GRDECL layout.

    Args:
        pfile (_XTgeoCFile): xtgeo file instance
        name (str, optional): Name of parameter. Defaults to "unknown".
        grid (Grid(), optional): XTGeo Grid instance. Defaults to None.

    Raises:
        xtgeo.KeywordNotFoundError: Cannot find property...
    """

    pfile.get_cfhandle()

    # scan file for properties; these have similar binary format as e.g. EGRID
    logger.info("Make kwlist by scanning")
    kwlist = utils.scan_keywords(
        pfile, fformat="xecl", maxkeys=1000, dataframe=False, dates=False
    )
    bpos = {}
    bpos[name] = -1

    for kwitem in kwlist:
        kwname, kwtype, kwlen, kwbyte = kwitem
        logger.info("KWITEM: %s", kwitem)
        if name == kwname:
            bpos[name] = kwbyte
            break

    if bpos[name] == -1:
        raise xtgeo.KeywordNotFoundError(
            "Cannot find property name {} in file {}".format(name, pfile.name)
        )
    self._ncol = grid.ncol
    self._nrow = grid.nrow
    self._nlay = grid.nlay

    values = _eclbin.eclbin_record(pfile, kwname, kwlen, kwtype, kwbyte)
    if kwtype == "INTE":
        self._isdiscrete = True
        # make the code list
        uniq = np.unique(values).tolist()
        codes = dict(zip(uniq, uniq))
        codes = {key: str(val) for key, val in codes.items()}  # val: strings
        self.codes = codes

    else:
        self._isdiscrete = False
        values = values.astype(np.float64)  # cast REAL (float32) to float64
        self.codes = {}

    # property arrays from binary GRDECL will be for all cells, but they
    # are in Fortran order, so need to convert...

    actnum = grid.get_actnum().values
    allvalues = values.reshape(self.dimensions, order="F")
    allvalues = np.asanyarray(allvalues, order="C")
    allvalues = ma.masked_where(actnum < 1, allvalues)
    self.values = allvalues
    self._name = name

    pfile.cfclose()


@contextmanager
def open_grdecl(grdecl_file, keywords):
    """
    Opens the grdecl_file for reading. Is a generator for tuples of keyword /
    values in records of that file. The format of the file must be that of the
    GRID section of a eclipse input DATA file.

    The records looked for must be "simple" ie.  start with the keyword, be
    followed by single word values and ended by a slash ('/').

    .. code-block:: none

        KEYWORD
        value value value /

    reading the above file with :code:`open_grdecl("filename.grdecl",
    keywords="KEYWORD")` will generate :code:`[("KEYWORD", ["value", "value",
    "value"])]`

    open_grdecl does not follow includes, obey skips, parse MESSAGE commands or
    make exception for groups and subrecords.

    Raises:
        ValueError: when end of file is reached without terminating a keyword.

    Args:
        keywords (List[str]): Which keywords to look for, these are expected to
        be at the start of a line in the file  and the respective values
        following on subsequent lines separated by whitespace. Reading of a
        keyword is completed by a final '\'. See example above.

    """

    def read_grdecl(grdecl_stream):
        words = []
        keyword = None

        line_no = 1
        line = grdecl_stream.readline()
        while line:
            if keyword is None:
                splitted = line.split()
                if splitted:
                    first_word = splitted[0]
                    if first_word in keywords:
                        keyword = first_word
                        logger.debug("Keyword %s found on line %d", keyword, line_no)

            else:
                for word in line.split():
                    if word == "--":
                        break

                    if word == "/":
                        yield (keyword, words)
                        words = []
                        keyword = None
                        break
                    words.append(word)
            line = grdecl_stream.readline()
            line_no += 1

        if keyword is not None:
            raise ValueError(f"Reached end of stream while reading {keyword}")

    with open(grdecl_file, "r") as stream:
        yield read_grdecl(stream)


def read_grdecl_3d_property(filename, keyword, dimensions, dtype=float):
    """
    Read a 3d grid property from a grdecl file, see open_grdecl for description
    of format.

    Args:
        filename (pathlib.Path or str): File in grdecl format.
        keyword (str): The keyword of the property in the file
        dimensions ((int,int,int)): Triple of the size of grid.
        dtype (function): The datatype to be read, ie., float.

    Raises:
        xtgeo.KeywordNotFoundError: If keyword is not found in the file.

    Returns:
        numpy array with given dimensions and data type read
        from the grdecl file.
    """
    result = None

    with open_grdecl(filename, keywords=(keyword,)) as kw_generator:
        try:
            _, result = next(kw_generator)
        except StopIteration as si:
            raise xtgeo.KeywordNotFoundError(
                f"Cannot import {keyword}, not present in file {filename}?"
            ) from si

    # The values are stored in F order in the grdecl file
    f_order_values = np.array([dtype(v) for v in result])
    return np.ascontiguousarray(f_order_values.reshape(dimensions, order="F"))


def import_grdecl_prop(self, pfile, name="unknown", grid=None):
    """Read a GRDECL ASCII property record"""

    if grid is None:
        raise ValueError("A grid instance is required as argument")

    self._ncol = grid.ncol
    self._nrow = grid.nrow
    self._nlay = grid.nlay
    self._name = name
    self._filesrc = pfile
    actnumv = grid.get_actnum().values

    self.values = ma.masked_where(
        actnumv == 0, read_grdecl_3d_property(pfile.file, name, self.dimensions, float)
    )
