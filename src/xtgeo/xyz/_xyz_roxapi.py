# coding: utf-8
"""Roxar API functions for XTGeo Points/Polygons"""
import os
import tempfile

import numpy as np
import pandas as pd

import xtgeo
from xtgeo.common import XTGeoDialog, _XTGeoFile
from xtgeo.roxutils import RoxUtils
from xtgeo.xyz import _xyz_io

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)

# pylint: disable=protected-access
VALID_STYPES = ["horizons", "zones", "clipboard", "general2d_data", "faults"]
VALID_STYPES_EXPORT = VALID_STYPES + ["horizon_picks"]


def _check_stypes_names_category(roxutils, stype, name, category, export=False):
    """General check of some input values."""
    stype = stype.lower()

    valid_stypes = VALID_STYPES_EXPORT if export else VALID_STYPES

    if stype not in valid_stypes:
        raise ValueError(
            f"Invalid stype value! For key <stype> the value {stype} is not supported, "
            f"legal stype values are: {valid_stypes}"
        )

    if not name:
        raise ValueError("The name is missing or empty.")

    logger.info("The stype is: %s", stype)
    if stype in ("horizons", "zones") and (name is None or not category):
        raise ValueError(
            "Need to spesify both name and category for horizons and zones"
        )

    # note: check of clipboard va Roxar API 1.2 is now removed as usage of
    # such old API versions is obsolute.
    if stype == "general2d_data" and not roxutils.version_required("1.6"):
        raise NotImplementedError(
            "API Support for general2d_data is missing in this RMS version"
            f"(current API version is {roxutils.roxversion} - required is 1.6"
        )


def import_xyz_roxapi(
    project, name, category, stype, realisation, attributes, is_polygons
):  # pragma: no cover
    """Import a Points or Polygons item via ROXAR API spec.

    'Import' means transfer of data from Roxar API memory space to XTGeo memory space.

    ~ a part of a classmethod, and it will return the following kwargs to __init__()::

        xname
        yname
        zname
        pname # optional for Polygons
        name
        dataframe
        values=None  # since dataframe is set
        filesrc
    """

    rox = RoxUtils(project, readonly=True)

    _check_stypes_names_category(rox, stype, name, category)

    if attributes and not rox.version_required("1.6"):
        result = _roxapi_import_xyz_viafile(
            rox, name, category, stype, realisation, is_polygons
        )
    else:
        result = _roxapi_import_xyz(
            rox, name, category, stype, realisation, is_polygons, attributes
        )

    rox.safe_close()
    return result


def _roxapi_import_xyz_viafile(
    rox, name, category, stype, realisation, is_polygons
):  # pragma: no cover
    """Read XYZ from file due to a missing feature in Roxar API wrt attributes.

    However, attributes will be present in Roxar API from RMS version 12, and this
    routine should be replaced!
    """

    try:
        import roxar  # pylint: disable=import-outside-toplevel
    except ImportError as err:
        raise ImportError(
            "roxar not available, this functionality is not available"
        ) from err

    if not _check_category_etc(rox.project, name, category, stype, realisation):
        raise RuntimeError(
            "It appears that name and or category is not present: "
            "name={}, category/folder={}, stype={}".format(name, category, stype)
        )

    rox_xyz = _get_roxxyz(
        rox,
        name,
        category,
        stype,
        mode="get",
        is_polygons=is_polygons,
    )

    kwargs = {}
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            logger.info("Made a tmp folder: %s", tmpdir)
            tfile = os.path.join(tmpdir, "generic.rmsattr")
            rox_xyz.save(tfile, roxar.FileFormat.RMS_POINTS)
            pfile = _XTGeoFile(tfile)
            kwargs = _xyz_io.import_rms_attr(pfile)

    except KeyError as kwe:
        logger.error(kwe)

    return kwargs


def _roxapi_import_xyz(
    rox, name, category, stype, realisation, is_polygons, attributes
):  # pragma: no cover
    """From RMS to XTGeo"""
    kwargs = {}

    if not _check_category_etc(rox.project, name, category, stype, realisation):
        raise RuntimeError(
            "It appears that name and or category is not present: "
            "name={}, category/folder={}, stype={}".format(name, category, stype)
        )

    kwargs["xname"] = "X_UTME"
    kwargs["yname"] = "Y_UTMN"
    kwargs["zname"] = "Z_TVDSS"

    if is_polygons:
        kwargs["pname"] = "POLY_ID"

    roxxyz = _get_roxxyz(
        rox,
        name,
        category,
        stype,
        mode="get",
        is_polygons=is_polygons,
    )

    values = _get_roxvalues(roxxyz, realisation=realisation)

    dfr = _roxapi_xyz_to_dataframe(values, is_polygons=is_polygons)

    # handling attributes for points, from Roxar API version 1.6
    if attributes and not is_polygons:
        attr_names = roxxyz.get_attributes_names(realisation=realisation)
        logger.info("XYZ attribute names are: %s", attr_names)
        attr_dict = _get_rox_attrvalues(roxxyz, attr_names, realisation=realisation)
        dfr, datatypes = _add_attributes_to_dataframe(dfr, attr_dict)
        kwargs["attributes"] = datatypes

    kwargs["values"] = dfr
    return kwargs


def _roxapi_xyz_to_dataframe(roxxyz, is_polygons=False):  # pragma: no cover
    """Transforming some XYZ from ROXAPI to a Pandas dataframe."""

    # In ROXAPI, polygons/polylines are a list of numpies, while
    # points is just a numpy array. Hence a polyg* may be identified
    # by being a list after import

    logger.info("Points/polygons/polylines from roxapi to xtgeo...")
    cnames = ["X_UTME", "Y_UTMN", "Z_TVDSS"]

    if is_polygons and isinstance(roxxyz, list):
        # polylines/-gons
        dfs = []
        for idx, poly in enumerate(roxxyz):
            dataset = pd.DataFrame.from_records(poly, columns=cnames)
            dataset["POLY_ID"] = idx
            dfs.append(dataset)

        dfr = pd.concat(dfs)

    elif not is_polygons and isinstance(roxxyz, np.ndarray):
        # points
        dfr = pd.DataFrame.from_records(roxxyz, columns=cnames)

    else:
        raise RuntimeError(f"Unknown error in getting data from Roxar: {type(roxxyz)}")

    dfr.reset_index(drop=True, inplace=True)
    return dfr


def _add_attributes_to_dataframe(dfr, attributes: dict):  # pragma: no cover
    """Add attributes to dataframe (points only) for Roxar API ver 1.6+"""

    logger.info("Attributes adding to dataframe...")
    newdfr = dfr.copy()

    datatypes = {}
    for name, values in attributes.items():
        dtype = str(values.dtype)
        if "int" in dtype:
            datatypes[name] = "int"
            values = np.ma.filled(values, fill_value=xtgeo.UNDEF_INT)
        elif "float" in dtype:
            datatypes[name] = "float"
            values = np.ma.filled(values, fill_value=xtgeo.UNDEF)
        else:
            datatypes[name] = "str"
            values = np.ma.filled(values, fill_value="UNDEF")

        newdfr[name] = values

    return newdfr, datatypes


def export_xyz_roxapi(
    self, project, name, category, stype, pfilter, realisation, attributes
):  # pragma: no cover
    """Export (store) a XYZ item from XTGeo to RMS via ROXAR API spec."""
    rox = RoxUtils(project, readonly=False)

    _check_stypes_names_category(rox, stype, name, category, export=True)

    if stype == "horizon_picks":
        _roxapi_export_xyz_hpicks(
            self, rox, name, category, stype, realisation, attributes
        )

    if attributes and not rox.version_required("1.6"):
        _roxapi_export_xyz_viafile(
            self, rox, name, category, stype, pfilter, realisation, attributes
        )
    else:
        _roxapi_export_xyz(
            self, rox, name, category, stype, pfilter, realisation, attributes
        )

    if rox._roxexternal:
        rox.project.save()

    rox.safe_close()


def _roxapi_export_xyz_hpicks(
    self, rox, name, category, stype, realisation, attributes
):  # pragma: no cover
    """
    Export/store as RMS horizon picks; this is only valid if points belong to wells
    """
    # need to think on design!
    raise NotImplementedError


def _roxapi_export_xyz_viafile(
    self, rox, name, category, stype, pfilter, realisation, attributes
):  # pragma: no cover
    """Set points/polys within RMS with attributes, using file workaround"""

    logger.warning("Realisation %s not in use", realisation)

    try:
        import roxar  # pylint: disable=import-outside-toplevel
    except ImportError as err:
        raise ImportError(
            "roxar not available, this functionality is not available"
        ) from err

    proj = rox.project

    if not _check_category_etc(proj, name, category, stype, realisation, mode="set"):
        raise RuntimeError("Cannot access correct category or name in RMS")

    roxxyz = _get_roxitem(self, proj, name, category, stype, mode="set")

    # make a temporary folder and work within the with.. block
    with tempfile.TemporaryDirectory() as tmpdir:
        logger.info("Made a tmp folder: %s", tmpdir)
        ncount = self.to_file(
            os.path.join(tmpdir, "generic.rmsattr"),
            fformat="rms_attr",
            pfilter=pfilter,
            attributes=attributes,
        )

        if ncount:
            roxxyz.load(
                os.path.join(tmpdir, "generic.rmsattr"), roxar.FileFormat.RMS_POINTS
            )


def _roxapi_export_xyz(
    self, rox, name, category, stype, pfilter, realisation, attributes
):  # pragma: no cover

    logger.warning("Realisation %s not in use", realisation)

    proj = rox.project
    if not _check_category_etc(proj, name, category, stype, realisation, mode="set"):
        raise RuntimeError("Cannot access correct category or name in RMS")

    roxxyz = _get_roxitem(self, proj, name, category, stype, mode="set")

    # pylint: disable=len-as-condition
    if self.dataframe is None or len(self.dataframe.index) == 0:
        return

    dfrcopy = self.dataframe.copy()
    # apply pfilter if any
    if pfilter:
        for key, val in pfilter.items():
            if key in dfrcopy.columns:
                dfrcopy = dfrcopy.loc[dfrcopy[key].isin(val)]
            else:
                raise KeyError(
                    "The requested pfilter key {} was not "
                    "found in dataframe. Valid keys are "
                    "{}".format(key, dfrcopy.columns)
                )

    if isinstance(self, xtgeo.Polygons):
        arrxyz = []
        polys = dfrcopy.groupby(self.pname)
        for _id, grp in polys:
            arr = np.stack([grp[self.xname], grp[self.yname], grp[self.zname]], axis=1)
            arrxyz.append(arr)
    else:
        xyz = dfrcopy
        arrxyz = np.stack([xyz[self.xname], xyz[self.yname], xyz[self.zname]], axis=1)

    if (
        isinstance(arrxyz, np.ndarray)
        and arrxyz.size == 0
        or isinstance(arrxyz, list)
        and len(arrxyz) == 0
    ):
        return

    roxxyz.set_values(arrxyz)

    if attributes and isinstance(self, xtgeo.Points) and len(self.dataframe) > 3:
        dfr = _cast_dataframe_attrs_to_numeric(dfrcopy)
        for name in dfr.columns[3:]:
            values = dfr[name].values
            if "float" in str(values.dtype):
                values = np.ma.masked_greater(values, xtgeo.UNDEF_LIMIT)
            elif "int" in str(values.dtype):
                values = np.ma.masked_greater(values, xtgeo.UNDEF_INT_LIMIT)
            else:
                # masking has no meaning for strings?
                values = values.astype(str)
                values = np.char.replace(values, "UNDEF", "")

            logger.info("Store Point attribute %s to Roxar API", name)
            roxxyz.set_attribute_values(name, values)


def _cast_dataframe_attrs_to_numeric(dfr):
    """Cast the attribute dataframe columns to numerical datatypes if possible.

    In some case, attribute columns get dtype 'object' while they clearly
    represents a numerical property (int or float). Here the pandas to_numerics()
    function is applied per attribute column, and will do a conversion if possible;
    otherwise the 'object' dtype will be preserved.
    """
    if len(dfr) <= 3:
        return dfr

    newdfr = dfr.copy()
    for name in dfr.columns[3:]:
        newdfr[name] = pd.to_numeric(dfr[name], errors="ignore")
    return newdfr


def _check_category_etc(
    proj, name, category, stype, realisation, mode="get"
):  # pylint: disable=too-many-branches  # pragma: no cover

    """Helper to check if valid placeholder' whithin RMS."""

    logger.warning("Realisation %s not in use", realisation)

    stypedict = {"horizons": proj.horizons, "zones": proj.zones, "faults": proj.faults}

    if stype in stypedict.keys():
        if name not in stypedict[stype]:
            logger.error("Cannot access name in stype=%s: %s", stype, name)
            return False
        if category not in stypedict[stype].representations:
            logger.error("Cannot access category in stype=%s: %s", stype, category)
            return False

    elif stype in ("clipboard", "general2d_data") and mode == "get":
        folders = None
        if category:
            if isinstance(category, list):
                folders = category
            elif isinstance(category, str) and "|" in category:
                folders = category.split("|")
            elif isinstance(category, str) and "/" in category:
                folders = category.split("/")
            elif isinstance(category, str):
                folders = []
                folders.append(category)
            else:
                raise RuntimeError(
                    "Cannot parse category: {}, see documentation!".format(category)
                )
            try:
                roxxyz = getattr(proj, stype).folders[folders]
            except KeyError as keyerr:
                logger.error(
                    "Cannot access clipboards folder (not existing?): %s", keyerr
                )
                return False
        else:
            roxxyz = proj.clipboard

        if name not in roxxyz.keys():
            raise ValueError("Name {} is not within Clipboard...".format(name))

    elif stype in ("clipboard", "general2d_data") and mode == "set":
        logger.info("No need to check clipboard while setting data")
    else:
        raise ValueError(f"Invalid stype: {stype}")

    return True


def _get_roxitem(self, proj, name, category, stype, mode="set"):  # pragma: no cover
    # pylint: disable=too-many-branches
    if stype == "horizons":
        roxxyz = proj.horizons[name][category]
    elif stype == "zones":
        roxxyz = proj.zones[name][category]
    elif stype == "faults":
        roxxyz = proj.faults[name][category]
    elif stype in ["clipboard", "general2d_data"]:
        folders = None
        roxxyz = getattr(proj, stype)
        if category:
            if isinstance(category, list):
                folders = category
            elif isinstance(category, str) and "|" in category:
                folders = category.split("|")
            elif isinstance(category, str) and "/" in category:
                folders = category.split("/")
            elif isinstance(category, str):
                folders = []
                folders.append(category)
            else:
                raise RuntimeError(
                    "Cannot parse category: {}, see documentation!".format(category)
                )

            if mode == "get":
                roxxyz = roxxyz.folders[folders]

        if mode == "get":
            roxxyz = roxxyz[name]

        elif mode == "set":

            # clipboard folders will be created if not present, and overwritten else
            if isinstance(self, xtgeo.Polygons):
                roxxyz = roxxyz.create_polylines(name, folders)
            else:
                roxxyz = roxxyz.create_points(name, folders)

    else:
        raise ValueError("Unsupported stype: {}".format(stype))

    return roxxyz


def _get_roxxyz(
    rox, name, category, stype, mode="set", is_polygons=False
):  # pragma: no cover
    # pylint: disable=too-many-branches
    """Get the correct rox_xyz which is some pointer to a RoxarAPI structure."""
    if stype == "horizons":
        rox_xyz = rox.project.horizons[name][category]
    elif stype == "zones":
        rox_xyz = rox.project.zones[name][category]
    elif stype == "faults":
        rox_xyz = rox.project.faults[name][category]
    elif stype in ["clipboard", "general2d_data"]:
        folders = None
        rox_xyz = getattr(rox.project, stype)
        if category:
            if isinstance(category, list):
                folders = category
                folders.append(category)
            elif isinstance(category, str) and "|" in category:
                folders = category.split("|")
            elif isinstance(category, str) and "/" in category:
                folders = category.split("/")
            elif isinstance(category, str):
                folders = []
                folders.append(category)
            else:
                raise RuntimeError(
                    f"Cannot parse category: {category}, see documentation!"
                )

            if mode == "get":
                rox_xyz = rox_xyz.folders[folders]

        if mode == "get":
            rox_xyz = rox_xyz[name]

        elif mode == "set":
            # clipboard folders will be created if not present, and overwritten else
            if is_polygons:
                rox_xyz = rox_xyz.create_polylines(name, folders)
            else:
                rox_xyz = rox_xyz.create_points(name, folders)

    else:
        raise TypeError(f"Unsupported stype: {stype}")  # shall never get this far...

    return rox_xyz


def _get_roxvalues(rox_xyz, realisation=0):  # pragma: no cover
    """Return primary values from the Roxar API, numpy (Points) or list (Polygons)."""
    try:
        roxitem = rox_xyz.get_values(realisation)
        logger.info(roxitem)
    except KeyError as kwe:
        logger.error(kwe)

    return roxitem


def _get_rox_attrvalues(rox_xyz, attrnames, realisation=0) -> dict:  # pragma: no cover
    """Return attribute values from the Roxar API, numpy (Points) or list (Polygons)."""

    roxitems = {}
    for attrname in attrnames:
        values = rox_xyz.get_attribute_values(attrname, realisation=realisation)
        roxitems[attrname] = values

    return roxitems
