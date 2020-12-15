# -*- coding: utf-8 -*-
"""Private import and export routines for XYZ stuff."""


import numpy as np
import pandas as pd
import xtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)

# -------------------------------------------------------------------------
# Import/Export methods for various formats
# Note: 'self' is the XYZ instance which may be Points/Polygons
# -------------------------------------------------------------------------


def import_xyz(self, pfile, zname="Z_TVDSS"):
    """Simple X Y Z file. All points as Pandas framework."""

    self.zname = zname

    self._df = pd.read_csv(
        pfile,
        delim_whitespace=True,
        skiprows=0,
        header=None,
        names=[self._xname, self._yname, zname],
        dtype=np.float64,
        na_values=999.00,
    )

    logger.debug(self._df.head())


def import_zmap(self, pfile, zname="Z_TVDSS"):
    """The zmap ascii polygon format; not sure about all details."""
    # ...seems that I just
    # take in the columns starting from @<blank> line as is.
    # Potential here to improve...

    #
    # !
    # ! File exported from RMS.
    # !
    # ! Project:
    # ! Date:          2017-11-07T17:22:30
    # !
    # ! Polygons/points Z-MAP file generated for ''.
    # ! Coordinate system is ''.
    # !
    # !------------------------------------------------------------------
    # @FREE POINT        , DATA, 80, 1
    # X (EASTING)        , 1, 1,  1,      1, 20,,    1.0E+30,,,   4, 0
    # Y (NORTHING)       , 2, 2,  1,     21, 40,,    1.0E+30,,,   4, 0
    # Z VALUE            , 3, 3,  1,     41, 60,,    1.0E+30,,,   4, 0
    # SEG I.D.           , 4, 35, 1,     61, 70,,    1.0E+30,,,   0, 0
    # @
    #    457357.781250      6782685.500000      1744.463379         0
    #    457359.343750      6782676.000000      1744.482056         0
    #    457370.906250      6782606.000000      1744.619507         0
    #    457370.468750      6782568.500000      1745.868286         0

    dtype = {
        self._xname: np.float64,
        self._yname: np.float64,
        zname: np.float64,
        self._pname: np.int32,
    }

    self._df = pd.read_csv(
        pfile,
        delim_whitespace=True,
        skiprows=16,
        header=None,
        names=[self._xname, self._yname, zname, self._pname],
        dtype=dtype,
        na_values=1.0e30,
    )

    logger.debug(self._df.head())


def import_rms_attr(self, pfile, zname="Z_TVDSS"):
    """The RMS ascii file with atttributes"""

    # Discrete  FaultBlock
    # String    FaultTag
    # Float     VerticalSep
    # 519427.941  6733887.914  1968.988    6  UNDEF  UNDEF
    # 519446.363  6732037.910  1806.782    19  UNDEF  UNDEF
    # 519446.379  6732137.910  1795.707    19  UNDEF  UNDEF

    self._zname = zname
    dtype = {self._xname: np.float64, self._yname: np.float64, self._zname: np.float64}

    names = [self._xname, self._yname, self._zname]

    # parse header
    skiprows = 0
    with open(pfile, "r") as rmsfile:
        for iline in range(20):
            fields = rmsfile.readline().split()
            if len(fields) != 2:
                skiprows = iline
                break

            dty, cname = fields
            dtyx = None
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
            self._attrs[cname] = dtyx

    self._df = pd.read_csv(
        pfile,
        delim_whitespace=True,
        skiprows=skiprows,
        header=None,
        names=names,
        dtype=dtype,
    )

    for col in self._df.columns[3:]:
        if col in self._attrs:
            if self._attrs[col] == "float":
                self._df[col].replace("UNDEF", xtgeo.UNDEF, inplace=True)
            elif self._attrs[col] == "int":
                self._df[col].replace("UNDEF", xtgeo.UNDEF_INT, inplace=True)


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
