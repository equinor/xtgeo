# -*- coding: utf-8 -*-
"""Private import and export routines for XYZ stuff."""

from collections import OrderedDict

import numpy as np
import pandas as pd
import xtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)

# -------------------------------------------------------------------------
# Import/Export methods for various formats; primarely class methods
# -------------------------------------------------------------------------


def import_xyz(pfile, zname="Z_TVDSS", is_polygons=False):
    """Simple text X Y Z files on ROFF_ascii or just XYZ.

    The are some subtle differences between a Roff text file and XYZ file
    according to RMS documentation:

    Points:

    roff_ascii and XYZ seems identical!

    Polygons:

    roff_ascii have 3 columns and will contain a line 999.0 999.0 999.0 to
    seperate polygons.

    The XYZ files valid for Polygons will have an ID column. It may be 3 or 4 columns,
    if only 3 then the Z values is assumed to be 0.0


    """
    assume_roff_txt = True  # assume that XYZ file is ROFF text based

    ncolumns = 3
    with open(pfile.file, "r") as stream:
        fields = stream.readline().split()
        if len(fields) == 4:
            assume_roff_txt = False
            ncolumns = 4
            if not is_polygons:
                raise TypeError(
                    "Points is assumed but this appears to be a XYZP "
                    "for Polygons as the file has 4 columns"
                )
        else:
            if not is_polygons:
                assume_roff_txt = True
            else:
                assume_roff_txt = False

            if "." in fields[2]:
                assume_roff_txt = True

    args = {}
    _xn = args["xname"] = "X_UTME"
    _yn = args["yname"] = "Y_UTMN"
    _zn = args["zname"] = zname
    _pn = args["pname"] = None

    if assume_roff_txt:
        dfr = pd.read_csv(
            pfile.file,
            delim_whitespace=True,
            skiprows=0,
            header=None,
            names=[_xn, _yn, _zn],
            dtype=np.float64,
            na_values=999.00,
        )
    if not is_polygons:
        dfr.dropna(inplace=True)

    if is_polygons and assume_roff_txt:
        # make a new polygon for every input line which are undefined (999 in input)
        _pn = args["pname"] = "POLY_ID"
        dfr[_pn] = dfr.isnull().all(axis=1).cumsum().dropna()
        dfr.dropna(axis=0, inplace=True)
        dfr.reset_index(inplace=True, drop=True)

    if is_polygons and not assume_roff_txt:
        _pn = args["pname"] = "POLY_ID"
        dtypes = {_xn: np.float64, _yn: np.float64, _pn: np.int32}
        if ncolumns == 4:
            dtypes = {_xn: np.float64, _yn: np.float64, _zn: np.float64, _pn: np.int32}

        dfr = pd.read_csv(
            pfile.file,
            delim_whitespace=True,
            skiprows=0,
            header=None,
            names=dtypes.keys(),
            dtype=dtypes,
            na_values=999.00,
        )
        if ncolumns == 3:
            dfr.insert(2, _pn, 0.0)

    args["is_polygons"] = is_polygons
    args["attributes"] = None
    args["dataframe"] = dfr
    args["values"] = None
    args["filesrc"] = pfile.name

    return args


def import_zmap(pfile, zname="Z_TVDSS", is_polygons=True):
    """The zmap ascii polygon format for Polygons.

    This format is only supported for Polygons in RMS. The ZMAP formats
    are, as always, difficult to get the grip of by googling as it seems that
    detailed standard is hard to find. This version supports what RMS
    does: 3 or 4 columns where last column is a line ID. If there
    are 3 columns then the Z values shall be 0.0. Data starts
    after the second @<blank> line as is.

    XTGeo will allow input as Points also if. Hence if is_polygons is False
    then the POLY_ID column will removed.

    Example::

      !------------------------------------------------------------------
      !
      ! File exported from RMS.
      !
      ! Project:
      ! Date:          2017-11-07T17:22:30
      !
      ! Polygons/points Z-MAP file generated for ''.
      ! Coordinate system is ''.
      !
      !------------------------------------------------------------------
      @FREE POINT        , DATA, 80, 1
      X (EASTING)        , 1, 1,  1,      1, 20,,    1.0E+30,,,   4, 0
      Y (NORTHING)       , 2, 2,  1,     21, 40,,    1.0E+30,,,   4, 0
      Z VALUE            , 3, 3,  1,     41, 60,,    1.0E+30,,,   4, 0
      SEG I.D.           , 4, 35, 1,     61, 70,,    1.0E+30,,,   0, 0
      @
         457357.781250      6782685.500000      1744.463379         0
         457359.343750      6782676.000000      1744.482056         0
         457370.906250      6782606.000000      1744.619507         0
         457370.468750      6782568.500000      1745.868286         0

    """

    args = {}
    _xn = args["xname"] = "X_UTME"
    _yn = args["yname"] = "Y_UTMN"
    _zn = args["zname"] = zname
    _pn = args["pname"] = "POLY_ID"

    # scan header
    skiprows = 0
    zcolumn = True  # if four columns, the third is Z
    with open(pfile.file, "r") as stream:
        count_a = 0  # number of '@' in first column
        for _ in range(999):
            fields = stream.readline().split()
            skiprows += 1
            if fields[0].startswith("@"):
                count_a += 1

            if count_a == 2:
                data = stream.readline().split()
                if len(data) == 3:
                    zcolumn = False
                break

    dtypes = {
        _xn: np.float64,
        _yn: np.float64,
        _zn: np.float64,
        _pn: np.int32,
    }

    cnames = [_xn, _yn, _zn, _pn]
    if not zcolumn:
        cnames = ([_xn, _yn, _pn],)
        dtypes = {
            _xn: np.float64,
            _yn: np.float64,
            _pn: np.int32,
        }

    dfr = pd.read_csv(
        pfile.file,
        delim_whitespace=True,
        skiprows=skiprows,
        header=None,
        names=cnames,
        dtype=dtypes,
        na_values=1.0e30,
    )
    if not zcolumn:
        dfr.insert(2, _zn, 0.0)  # inject as third column with values 0.0

    if not is_polygons:
        dfr.drop(_pn, axis=1, inplace=True)  # allow Points, remove ID column

    args["is_polygons"] = is_polygons
    args["attributes"] = None
    args["dataframe"] = dfr
    args["values"] = None
    args["filesrc"] = pfile.name

    return args


def import_rms_attr(pfile, zname="Z_TVDSS", is_polygons=False):
    """The RMS ascii file Points format with attributes.

    It appears that the the RMS attributes format is supported for Points only.
    Example::

       Discrete  FaultBlock
       String    FaultTag
       Float     VerticalSep
       519427.941  6733887.914  1968.988    6  UNDEF  UNDEF
       519446.363  6732037.910  1806.782    19  UNDEF  UNDEF
       519446.379  6732137.910  1795.707    19  UNDEF  UNDEF
    """

    if is_polygons is True:
        raise TypeError("The RMS attribute format is not supported for Polygons")

    args = {}
    _xn = args["xname"] = "X_UTME"
    _yn = args["yname"] = "Y_UTMN"
    _zn = args["zname"] = zname

    dtypes = {_xn: np.float64, _yn: np.float64, _zn: np.float64}
    all_dtypes = dtypes.copy()

    names = list(dtypes.keys())
    _attrs = OrderedDict()

    # parse header
    skiprows = 0
    with open(pfile.file, "r") as rmsfile:
        for iline in range(999):
            fields = rmsfile.readline().split()
            if len(fields) != 2:
                skiprows = iline
                break

            dty, cname = fields
            dtyx = None

            # note that Pandas treats dtype str as object, cf:
            # https://stackoverflow.com/questions/34881079
            if dty == "Discrete":
                dtyx = "int"
            elif dty == "String":
                dtyx = "str"
            elif dty == "Float":
                dtyx = "float"
            elif dty == "Int":
                dtyx = "int"
            else:
                dtyx = "str"
            names.append(cname)
            all_dtypes[cname] = dtyx
            _attrs[cname] = dtyx

    dfr = pd.read_csv(
        pfile.file,
        delim_whitespace=True,
        skiprows=skiprows,
        header=None,
        names=names,
        dtype=dtypes,
    )

    # handle undefined:
    for col in dfr.columns[3:]:
        if col in _attrs:
            if _attrs[col] == "float":
                dfr[col].replace("UNDEF", xtgeo.UNDEF, inplace=True)
            elif _attrs[col] == "int":
                dfr[col].replace("UNDEF", xtgeo.UNDEF_INT, inplace=True)
        dfr[col] = dfr[col].astype(all_dtypes[col])

    args["dataframe"] = dfr
    args["values"] = None
    args["attributes"] = _attrs
    args["is_polygons"] = False

    return args


def export_rms_attr(self, pfile, attributes=True, pfilter=None):
    """Export til RMS attribute, also called RMS extended set.

    If attributes is None, then it will be a simple XYZ file.

    Attributes can be a bool or a list. If True, then use all attributes.

    Filter is on the form {TopName: ['Name1', 'Name2']}

    Returns:
        The number of values exported. If value is 0; then no file
        is made.
    """

    df = self.dataframe.copy()

    if not df.index.any():
        logger.warning("Nothing to export")
        return 0

    columns = [self._xname, self._yname, self.zname]
    df.fillna(value=999.0, inplace=True)

    mode = "w"

    transl = {"int": "Discrete", "float": "Float", "str": "String"}

    logger.info("Attributes is %s", attributes)

    # apply pfilter if any
    if pfilter:
        for key, val in pfilter.items():
            if key in df.columns:
                df = df.loc[df[key].isin(val)]
            else:
                raise KeyError(
                    "The requested pfilter key {} was not "
                    "found in dataframe. Valid keys are "
                    "{}".format(key, df.columns)
                )

    if not attributes and self._pname in df.columns and self._ispolygons:
        # need to convert the dataframe
        df = _convert_idbased_xyz(self, df)

    elif attributes is True:
        attributes = list(self._attrs.keys())
        logger.info("Use all attributes: %s", attributes)

        for column in (self._xname, self._yname, self._zname):
            try:
                attributes.remove(column)
            except ValueError:
                continue

    if isinstance(attributes, list):
        mode = "a"
        columns += attributes
        with open(pfile, "w") as fout:
            for col in attributes:
                if col in df.columns:
                    fout.write(transl[self._attrs[col]] + " " + col + "\n")
                    if self._attrs[col] == "int":
                        df[col].replace(xtgeo.UNDEF_INT, "UNDEF", inplace=True)
                    elif self._attrs[col] == "float":
                        df[col].replace(xtgeo.UNDEF, "UNDEF", inplace=True)

    with open(pfile, mode) as fc:
        df.to_csv(
            fc, sep=" ", header=None, columns=columns, index=False, float_format="%.3f"
        )

    return len(df.index)


def _convert_idbased_xyz(self, df):
    """Conversion of format from ID column to 999 flag."""

    # If polygons, there is a 4th column with POLY_ID. This needs
    # to replaced by adding 999 line instead (for polygons)
    # prior to XYZ export or when interactions in CXTGEO

    idgroups = df.groupby(self._pname)

    newdf = pd.DataFrame(columns=[self._xname, self._yname, self._zname])
    udef = pd.DataFrame(
        [[999.0, 999.0, 999.0]], columns=[self._xname, self._yname, self._zname]
    )

    for _id, gr in idgroups:
        dfx = gr.drop(self._pname, axis=1)
        newdf = newdf.append([dfx, udef], ignore_index=True)

    return newdf


def export_rms_wpicks(self, pfile, hcolumn, wcolumn, mdcolumn="M_MDEPTH"):
    """Export til RMS wellpicks

    If a MD column (mdcolumn) exists, it will use the MD

    Args:
        pfile (str): File to export to
        hcolumn (str): Name of horizon/zone column in the point set
        wcolumn (str): Name of well column in the point set
        mdcolumn (str): Name of measured depht column (if any)
    Returns:
        The number of values exported. If value is 0; then no file
        is made.

    """

    df = self.dataframe.copy()

    if not df.index.any():
        logger.warning("Nothing to export")
        return 0

    columns = []

    if hcolumn in df.columns:
        columns.append(hcolumn)
    else:
        raise ValueError(
            "Column for horizons/zones <{}> " "not present".format(hcolumn)
        )

    if wcolumn in df.columns:
        columns.append(wcolumn)
    else:
        raise ValueError("Column for wells <{}> " "not present".format(wcolumn))

    if mdcolumn in df.columns:
        columns.append(mdcolumn)
    else:
        columns += [self._xname, self._yname, self._zname]

    if not df.index.any():
        logger.warning("Nothing to export")
        return 0

    with open(pfile, "w") as fc:
        df.to_csv(fc, sep=" ", header=None, columns=columns, index=False)

    return len(df.index)
