# -*- coding: utf-8 -*-
"""Private import and export routines for XYZ stuff."""

from collections import OrderedDict

import numpy as np
import pandas as pd
import xtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)


def import_xyz(pfile, zname="Z_TVDSS"):
    """Simple X Y Z file. All points as Pandas framework."""
    return {
        "zname": zname,
        "xname": "X_UTME",
        "yname": "Y_UTMN",
        "values": pd.read_csv(
            pfile.file,
            delim_whitespace=True,
            skiprows=0,
            header=None,
            names=["X_UTME", "Y_UTMN", zname],
            dtype=np.float64,
            na_values=999.00,
        ),
    }


def import_zmap(pfile, zname="Z_TVDSS"):
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

    xname = "X_UTME"
    yname = "Y_UTMN"
    pname = "POLY_ID"

    dtype = {
        xname: np.float64,
        yname: np.float64,
        zname: np.float64,
        pname: np.int32,
    }

    df = pd.read_csv(
        pfile.file,
        delim_whitespace=True,
        skiprows=16,
        header=None,
        names=[xname, yname, zname, pname],
        dtype=dtype,
        na_values=1.0e30,
    )

    return {"xname": xname, "yname": yname, "zname": zname, "values": df}


def import_rms_attr(pfile, zname="Z_TVDSS"):
    """The RMS ascii file Points format with attributes.

    It appears that the the RMS attributes format is supported for Points only,
    hence Polygons is not admitted.

    Example::

       Discrete  FaultBlock
       String    FaultTag
       Float     VerticalSep
       519427.941  6733887.914  1968.988     6  UNDEF  UNDEF
       519446.363  6732037.910  1806.782    19  UNDEF  UNDEF
       519446.379  6732137.910  1795.707    19  UNDEF  UNDEF

    Returns a kwargs list with the following items:
        xname
        yname
        zname
        values as a valid dataframe
        attributes

    Important notes from RMS manual and reverse engineering:

    * For discrete numbers use 'Discrete' or 'Integer', not 'Int'
    * For Discrete/Integer/Float both UNDEF and -999 will mark as undefined
    * For Discrete/Integer, numbers less than -999 seems to accepted by RMS
    * For String, use UNDEF only as undefined
    """

    kwargs = {}
    _xn = kwargs["xname"] = "X_UTME"
    _yn = kwargs["yname"] = "Y_UTMN"
    _zn = kwargs["zname"] = zname

    dtypes = {_xn: np.float64, _yn: np.float64, _zn: np.float64}

    names = list(dtypes.keys())
    _attrs = OrderedDict()

    # parse header
    skiprows = 0
    with open(pfile.file, "r") as rmsfile:
        for iline in range(20):
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
            _attrs[cname] = dtyx

    dfr = pd.read_csv(
        pfile.file,
        delim_whitespace=True,
        skiprows=skiprows,
        header=None,
        names=names,
        dtype=dtypes,
    )
    for col in dfr.columns[3:]:
        if col in _attrs:
            if _attrs[col] == "float":
                dfr[col].replace("UNDEF", xtgeo.UNDEF, inplace=True)
            elif _attrs[col] == "int":
                dfr[col].replace("UNDEF", xtgeo.UNDEF_INT, inplace=True)
            # cast to numerical if possible
        dfr[col] = pd.to_numeric(dfr[col], errors="ignore")

    kwargs["values"] = dfr
    kwargs["attributes"] = _attrs

    return kwargs


def to_file(
    xyz,
    pfile,
    fformat="xyz",
    attributes=False,
    pfilter=None,
    wcolumn=None,
    hcolumn=None,
    mdcolumn="M_MDEPTH",
    ispolygons=False,
    **kwargs,
):  # pylint: disable=redefined-builtin
    """Export XYZ (Points/Polygons) to file.

    Args:
        pfile (str): Name of file
        fformat (str): File format xyz/poi/pol / rms_attr /rms_wellpicks
        attributes (bool or list): List of extra columns to export (some formats)
            or True for all attributes present
        pfilter (dict): Filter on e.g. top name(s) with keys TopName
            or ZoneName as {'TopName': ['Top1', 'Top2']}
        wcolumn (str): Name of well column (rms_wellpicks format only)
        hcolumn (str): Name of horizons column (rms_wellpicks format only)
        mdcolumn (str): Name of MD column (rms_wellpicks format only)

    Returns:
        Number of points exported

    Note that the rms_wellpicks will try to output to:

    * HorizonName, WellName, MD  if a MD (mdcolumn) is present,
    * HorizonName, WellName, X, Y, Z  otherwise

    Raises:
        KeyError if pfilter is set and key(s) are invalid

    """
    filter_deprecated = kwargs.get("filter", None)
    if filter_deprecated is not None and pfilter is None:
        pfilter = filter_deprecated

    pfile = xtgeo._XTGeoFile(pfile)
    pfile.check_folder(raiseerror=OSError)

    ncount = 0
    if xyz.dataframe is None:
        logger.warning("Nothing to export!")
        return ncount

    if fformat is None or fformat in ["xyz", "poi", "pol"]:
        # NB! reuse export_rms_attr function, but no attributes
        # are possible
        ncount = export_rms_attr(
            xyz, pfile.name, attributes=False, pfilter=pfilter, ispolygons=ispolygons
        )

    elif fformat == "rms_attr":
        ncount = export_rms_attr(
            xyz,
            pfile.name,
            attributes=attributes,
            pfilter=pfilter,
            ispolygons=ispolygons,
        )
    elif fformat == "rms_wellpicks":
        ncount = export_rms_wpicks(xyz, pfile.name, hcolumn, wcolumn, mdcolumn=mdcolumn)

    if ncount is None:
        ncount = 0

    if ncount == 0:
        logger.warning("Nothing to export!")

    return ncount


def export_rms_attr(self, pfile, attributes=True, pfilter=None, ispolygons=False):
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

    if ispolygons:
        if not attributes and self._pname in df.columns:
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
        df.to_csv(fc, sep=" ", header=None, columns=columns, index=False)

    return len(df.index)


def _convert_idbased_xyz(self, df):
    """Conversion of format from ID column to 999 flag."""

    # If polygons, there is a 4th column with POLY_ID. This needs
    # to replaced by adding 999 line instead (for polygons)
    # prior to XYZ export or when interactions in CXTGEO

    idgroups = df.groupby(self._pname)

    newdf = pd.DataFrame(
        columns=[self._xname, self._yname, self._zname], dtype="float64"
    )
    udef = pd.DataFrame(
        [[999.0, 999.0, 999.0]], columns=[self._xname, self._yname, self._zname]
    )

    for _id, gr in idgroups:
        dfx = gr.drop(self._pname, axis=1)
        newdf = pd.concat([newdf, dfx, udef], ignore_index=True)

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


def _from_list_like(plist, zname, attrs, is_polygons) -> pd.DataFrame:
    """Import Points or Polygons from a list-like input.

    The following 'list-like' inputs are possible:

    * List of tuples [(x1, y1, z1, <id1>), (x2, y2, z2, <id2>), ...].
    * List of lists  [[x1, y1, z1, <id1>], [x2, y2, z2, <id2>], ...].
    * List of numpy arrays  [nparr1, nparr2, ...] where nparr1 is first row.
    * A numpy array with shape [nrow, ncol], where ncol >= 3
    * An existing pandas dataframe

    Points scenaria:
    * 3 columns, X Y Z
    * 4 or more columns: rest columns are attributes

    Polygons scenaria:
    * 3 columns, X Y Z. Here P column is assigned 0 afterwards
    * 4 or more columns:
        - if totnum = lenattrs + 3 then POLY_ID is missing and will be made
        - if totnum = lenattrs + 4 then assume that 4'th column is POLY_ID

    It is currently not much error checking that lists/tuples are consistent, e.g.
    if there always is either 3 or 4 elements per tuple, or that 4 number is
    an integer.

    Args:
        plist (str): List of tuples, each tuple is length 3 or 4
        zname (str): Name of third column
        attrs (dict): Attributes, for Points
        is_polygons (bool): Flag for Points or Polygons

    Returns:
        A valid datafram

    Raises:
        ValueError: If something is wrong with input

    .. versionadded:: 2.16
    """

    dfr = None
    if isinstance(plist, list):
        plist = np.array(plist)

    if isinstance(plist, np.ndarray):
        logger.info("Process numpy to points")
        if len(plist) == 0:
            return pd.DataFrame([], columns=["X_UTME", "Y_UTMN", zname])

        if plist.ndim != 2:
            raise ValueError("Input numpy array must two-dimensional")
        totnum = plist.shape[1]
        lenattrs = len(attrs) if attrs is not None else 0
        attr_first_col = 3
        if totnum == 3 + lenattrs:
            dfr = pd.DataFrame(plist[:, :3], columns=["X_UTME", "Y_UTMN", zname])
            dfr = dfr.astype(float)
            if is_polygons:
                # pname column is missing but assign 0 as ID
                dfr["POLY_ID"] = 0

        elif totnum == 4 + lenattrs and is_polygons:
            dfr = pd.DataFrame(
                plist[:, :4],
                columns=["X_UTME", "Y_UTMN", zname, "POLY_ID"],
            )
            attr_first_col = 4
        else:
            raise ValueError(
                f"Wrong length detected of row: {totnum}. "
                "Are attributes set correct?"
            )
        dfr.dropna()
        dfr = dfr.astype(np.float64)
        if is_polygons:
            dfr["POLY_ID"] = dfr["POLY_ID"].astype(np.int32)

        if lenattrs > 0:
            for enum, (key, dtype) in enumerate(attrs.items()):
                dfr[key] = plist[:, attr_first_col + enum]
                dfr = dfr.astype({key: dtype})

    else:
        raise TypeError("Not possible to make XYZ from given input")

    return dfr
