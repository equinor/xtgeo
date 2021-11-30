# -*- coding: utf-8 -*-
"""Private import and export routines for XYZ stuff."""

import warnings
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import pandas as pd
import xtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)


class _ValidDataFrame(pd.DataFrame):
    ...


# -------------------------------------------------------------------------
# Import/Export methods for various formats; primarely class methods
# -------------------------------------------------------------------------


def import_xyz(pfile, zname="Z_TVDSS", is_polygons=False):
    """Simple text X Y Z files on rms_ascii/irap_text or just XYZ.

    The format names 'irap_ascii', 'irap_text', 'rms_ascii', 'rms_text' are all the
    same format (the name has changed through history; the original was "irap_ascii"),
    while XYZ differs somewhat.

    The are some subtle differences between a RMS text file and XYZ file
    according to RMS documentation:

    Points:

    rms_ascii and XYZ seems identical!

    Polygons:

    rms_ascii have 3 columns and will contain a line 999.0 999.0 999.0 to
    seperate polygons.

    The XYZ files valid for Polygons will have an ID column. It may be 3 or 4 columns,
    if only 3 then the Z values is assumed to be 0.0. The last column is always the ID.
    """
    assume_rms_text = True  # assume that XYZ file is irap text based

    ncolumns = 3
    with open(pfile.file, "r", encoding="utf8") as stream:
        fields = stream.readline().split()
        if len(fields) == 4:
            assume_rms_text = False
            ncolumns = 4
            if not is_polygons:
                raise TypeError(
                    "Points is assumed but this appears to be a XYZP "
                    "for Polygons as the file has 4 columns"
                )
        else:
            if not is_polygons:
                assume_rms_text = True
            else:
                assume_rms_text = False

            if "." in fields[2]:
                assume_rms_text = True

    kwargs = {}
    _xn = kwargs["xname"] = "X_UTME"
    _yn = kwargs["yname"] = "Y_UTMN"
    _zn = kwargs["zname"] = zname

    if assume_rms_text:  # always True for Points
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

    # pylint: disable=unsupported-assignment-operation
    if is_polygons and assume_rms_text:
        # make a new polygon for every input line which are undefined (999 in input)
        _pn = kwargs["pname"] = "POLY_ID"
        dfr[_pn] = dfr.isnull().all(axis=1).cumsum().dropna()
        dfr.dropna(axis=0, inplace=True)
        dfr.reset_index(inplace=True, drop=True)

    if is_polygons and not assume_rms_text:
        _pn = kwargs["pname"] = "POLY_ID"
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

    kwargs["values"] = _ValidDataFrame(dfr)
    kwargs["filesrc"] = pfile.name

    return kwargs


def import_zmap(pfile, zname="Z_TVDSS", is_polygons=True):
    """The zmap ascii polygon format for Polygons.

    This format is only supported for Polygons in RMS but XTGeo allow Points also. The
    ZMAP formats are, as always, difficult to get the grip of by googling as it seems
    that detailed standard is hard to find. This version supports what RMS does: 3 or 4
    columns where last column is a line ID. If there are 3 columns then the Z values
    shall be 0.0. Data starts after the second @<blank> line as is.

    XTGeo will allow input as Points also if 'is_polygons' is False. Then the
    POLY_ID column will removed.

    Example::

      !------------------------------------------------------------------
      !
      ! File exported from RMS.
      !
      ! Project: ! Date:          2017-11-07T17:22:30
      !
      ! Polygons/points Z-MAP file generated for ''. ! Coordinate system is ''.
      !
      !------------------------------------------------------------------
      @FREE POINT        , DATA, 80, 1 X (EASTING)        , 1, 1,  1,      1, 20,,
      1.0E+30,,,   4, 0 Y (NORTHING)       , 2, 2,  1,     21, 40,,    1.0E+30,,,   4, 0
      Z VALUE            , 3, 3,  1,     41, 60,,    1.0E+30,,,   4, 0 SEG I.D. , 4, 35,
      1,     61, 70,,    1.0E+30,,,   0, 0
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
    with open(pfile.file, "r", encoding="utf8") as stream:
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
        dfr.drop(_pn, axis=1, inplace=True)
        del args["pname"]  # allow Points, remove ID column

    args["attributes"] = None
    args["values"] = _ValidDataFrame(dfr)
    args["filesrc"] = pfile.name

    return args


def import_rms_attr(pfile, zname="Z_TVDSS", is_polygons=False):
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
    logger.debug("The is_polygons flag is not applied here: %s", is_polygons)

    kwargs = {}
    _xn = kwargs["xname"] = "X_UTME"
    _yn = kwargs["yname"] = "Y_UTMN"
    _zn = kwargs["zname"] = zname

    dtypes = {_xn: np.float64, _yn: np.float64, _zn: np.float64}
    all_dtypes = dtypes.copy()

    names = list(dtypes.keys())
    _attrs = OrderedDict()

    # parse header
    skiprows = 0
    with open(pfile.file, "r", encoding="utf8") as rmsfile:
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
            elif dty == "Integer":
                dtyx = "int"
            elif dty == "Int":  # is needed for backward compatibility?
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

    # pylint: disable=unsubscriptable-object, unsupported-assignment-operation
    # handle undefined:
    undef_detected = False
    for col in dfr.columns[3:]:
        if col in _attrs:
            detect_undef = "UNDEF" in dfr[col].values or -999 in dfr[col].values
            if _attrs[col] == "float":
                dfr[col].replace("UNDEF", xtgeo.UNDEF, inplace=True)
                dfr[col].replace(-999.0, xtgeo.UNDEF, inplace=True)
            elif _attrs[col] == "int":
                dfr[col].replace("UNDEF", xtgeo.UNDEF_INT, inplace=True)
                dfr[col].replace(-999, xtgeo.UNDEF_INT, inplace=True)
        dfr[col] = dfr[col].astype(all_dtypes[col])
        if detect_undef:
            undef_detected = True

    if undef_detected:
        warnings.warn(
            "Undefined values are detected in attribute columns from input! "
            f"They are currently replaced by a large number which is "
            f"{xtgeo.UNDEF_INT} for integer entries and {xtgeo.UNDEF} "
            f"for float entries, while String entries will have UNDEF. "
            "This will change in version 3 of XTGeo to `pandas.NA`",
            FutureWarning,
        )

    kwargs["values"] = _ValidDataFrame(dfr)
    kwargs["attributes"] = _attrs

    return kwargs


def to_generic_file(
    self,
    pfile,
    fformat="xyz",
    attributes=False,
    pfilter=None,
    wcolumn=None,
    hcolumn=None,
    mdcolumn="M_MDEPTH",
    **kwargs,
):  # pylint: disable=redefined-builtin
    """Generic export to file."""
    filter_deprecated = kwargs.get("filter", None)
    if filter_deprecated is not None and pfilter is None:
        pfilter = filter_deprecated

    pfile = xtgeo._XTGeoFile(pfile)
    pfile.check_folder(raiseerror=OSError)

    if self.dataframe is None:
        ncount = 0
        logger.warning("Nothing to export!")
        return ncount

    if fformat is None or fformat in ["xyz", "poi", "pol"]:
        # NB! reuse export_rms_attr function, but no attributes
        # are possible
        ncount = export_rms_attr(self, pfile.name, attributes=False, pfilter=pfilter)

    elif fformat == "rms_attr":
        ncount = export_rms_attr(
            self, pfile.name, attributes=attributes, pfilter=pfilter
        )
    elif fformat == "rms_wellpicks":
        ncount = export_rms_wpicks(
            self, pfile.name, hcolumn, wcolumn, mdcolumn=mdcolumn
        )

    if ncount is None:
        ncount = 0

    if ncount == 0:
        logger.warning("Nothing to export!")

    return ncount


def export_rms_attr(self, pfile, attributes=True, pfilter=None):
    """Export til RMS attribute, also called RMS extended set.

    This format is only supported for Points. If attributes is None, then it will
    become a simple XYZ file.

    Attributes can be a bool or a list. If True, then use all known attributes.

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

    if attributes is True:
        attributes = list(self.attributes.keys())
        logger.info("Use all attributes: %s", attributes)

        for column in (self._xname, self._yname, self._zname):
            try:
                attributes.remove(column)
            except ValueError:
                continue

    if isinstance(attributes, list):
        mode = "a"
        columns += attributes
        with open(pfile, "w", encoding="utf8") as fout:
            for col in attributes:
                if col in df.columns:
                    fout.write(transl[self.attributes[col]] + " " + col + "\n")
                    if self.attributes[col] == "int":
                        df[col].replace(xtgeo.UNDEF_INT, "UNDEF", inplace=True)
                    elif self.attributes[col] == "float":
                        df[col].replace(xtgeo.UNDEF, "UNDEF", inplace=True)

    with open(pfile, mode, encoding="utf8") as fc:
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

    with open(pfile, "w", encoding="utf8") as fc:
        df.to_csv(fc, sep=" ", header=None, columns=columns, index=False)

    return len(df.index)


def initialize_by_values(values, zname, attrs, is_polygons):
    """Worker for values initialization for Points/Polygons."""
    # process values if present
    dfr = None
    filesrc = None

    if values is not None:

        if isinstance(values, _ValidDataFrame):
            dfr = values
        elif isinstance(values, (list, np.ndarray, pd.DataFrame)):
            # make instance from a list(-like) of 3 or 4 tuples or similar
            logger.info("Instance from list, np array or dataframe")

            dfr, filesrc = _from_list_like(values, zname, attrs, is_polygons)
        else:
            raise ValueError(
                f"The values are of type {type(values)} which is not supported."
            )

    return dfr, filesrc


def _from_list_like(plist, zname, attrs, is_polygons):
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
        A valid dataframe and filesrc

    Raises:
        ValueError: If something is wrong with input

    .. versionadded:: 2.16
    """

    filesrc = "Derived from: numpy array"
    dfr = None
    if isinstance(plist, list):
        # convert list/tuples to a 2D numpy and process the numpy below
        logger.info("Input list-like is a list, convert to a numpy...")
        try:
            plist = np.array(plist)
        except Exception as exc:
            warnings.warn(f"Cannot convert list to numpy array: {str(exc)}")
            raise
        filesrc = "Derived from: list input"

    if isinstance(plist, pd.DataFrame):
        # convert input dataframe to a 2D numpy and process the numpy below
        plist = plist.apply(deepcopy).to_numpy(copy=True)
        filesrc = "Derived from: dataframe input"

    if isinstance(plist, np.ndarray):
        logger.info("Process numpy to points")
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

    dfr = _ValidDataFrame(dfr)

    return dfr, filesrc
