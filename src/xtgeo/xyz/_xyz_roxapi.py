# coding: utf-8
"""Roxar API functions for XTGeo Points/Polygons"""
import os
import tempfile
import numpy as np
import pandas as pd

from xtgeo.common import XTGeoDialog
from xtgeo.roxutils import RoxUtils

from xtgeo.xyz import _xyz_io

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)

# default column names
DEFAULTNAME = {"x": "X_UTME", "y": "Y_UTMN", "z": "Z_TVDSS", "p": "POLY_ID"}

# pylint: disable=protected-access

# ======================================================================================
# roxapi -> xtgeo classmethod
# ======================================================================================


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

    rox = RoxUtils(project)

    if stype == "clipboard" and not rox.version_required("1.2"):
        minimumrms = rox.rmsversion("1.2")
        msg = (
            "Not supported in this ROXAPI version. Points/polygons access "
            "to clipboard requires RMS {}".format(minimumrms)
        )
        raise NotImplementedError(msg)

    if attributes:
        logger.info("XYZ with attributes")
        kwargs = _roxapi_import_xyz_viafile(
            rox, name, category, stype, realisation, is_polygons
        )
    else:
        logger.info("XYZ without attributes")
        kwargs = _roxapi_import_xyz(
            rox, name, category, stype, realisation, is_polygons
        )

    rox.safe_close()

    kwargs["name"] = name
    kwargs["filesrc"] = "RMS: {} ({})".format(name, category)
    kwargs["values"] = None

    return kwargs


def _get_roxar():
    try:
        import roxar  # pylint: disable=import-outside-toplevel
    except ImportError as err:
        raise ImportError(
            "roxar not available, this functionality is not available"
        ) from err
    return roxar


def _roxapi_import_xyz_viafile(
    rox, name, category, stype, realisation, is_polygons
):  # pragma: no cover
    """Read XYZ from file due to amissing feature in Raoxar API wrt attributes.

    However, attributes will be present in Roxar API from RMS version 12, and this
    routine should be replaced!
    """

    roxar = _get_roxar()

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

    args = None
    try:
        # make a temporary folder
        with tempfile.TemporaryDirectory() as tmpdir:
            logger.info("Made a tmp folder: %s", tmpdir)
            tfile = os.path.join(tmpdir, "generic.rmsattr")
            rox_xyz.save(tfile, roxar.FileFormat.RMS_POINTS)
            args = _xyz_io.import_rms_attr(tfile, is_polygons=is_polygons)

    except KeyError as kwe:
        logger.error(kwe)

    return args


def _roxapi_import_xyz(
    rox, name, category, stype, realisation, is_polygons
):  # pragma: no cover
    """From RMS Roxar API to XTGeo, will be a class method."""
    args = {}

    if not _check_category_etc(rox, name, category, stype, realisation):
        raise RuntimeError(
            "It appears that name and or category is not present: "
            "name={}, category/folder={}, stype={}".format(name, category, stype)
        )

    args["xname"] = DEFAULTNAME["x"]
    args["yname"] = DEFAULTNAME["y"]
    args["zname"] = DEFAULTNAME["z"]
    if is_polygons:
        args["pname"] = DEFAULTNAME["p"]

    roxitem = _get_roxxyz(
        rox,
        name,
        category,
        stype,
        mode="get",
        is_polygons=is_polygons,
    )

    values = _get_roxvalues(roxitem, realisation=realisation)
    args["dataframe"] = _roxapi_xyz_to_dataframe(values, is_polygons=is_polygons)

    return args


def _roxapi_xyz_to_dataframe(roxitem, is_polygons=False):  # pragma: no cover
    """Tranforming some XYZ from ROXAPI to a Pandas dataframe."""

    # In ROXAPI, polygons/polylines are a list of numpies, while
    # points is just a numpy array. Hence a polyg* may be identified
    # by being a list after import

    logger.info("Points/polygons/polylines from roxapi to xtgeo...")
    cnames = [DEFAULTNAME["x"], DEFAULTNAME["y"], DEFAULTNAME["z"]]

    if is_polygons and isinstance(roxitem, list):
        # polylines/-gons
        dfs = []
        for idx, poly in enumerate(roxitem):
            dataset = pd.DataFrame.from_records(poly, columns=cnames)
            dataset[DEFAULTNAME["p"]] = idx
            dfs.append(dataset)

        dfr = pd.concat(dfs)

    elif not is_polygons and isinstance(roxitem, np.ndarray):
        # points
        dfr = pd.DataFrame.from_records(roxitem, columns=cnames)

    else:
        raise RuntimeError(f"Unknown error in getting data from Roxar: {type(roxitem)}")

    dfr.reset_index(inplace=True)
    return dfr


# ======================================================================================
# xtgeo -> roxapi
# The .to_roxar is an instance method, and xyzpp refers to the instance: self
# ======================================================================================


def export_xyz_roxapi(
    xyzpp, project, name, category, stype, pfilter, realisation, attributes, is_polygons
):  # pragma: no cover
    """Export (store) a XYZ item from XTGeo to RMS via ROXAR API spec."""

    rox = RoxUtils(project)

    if stype == "clipboard" and not rox.version_required("1.2"):
        minimumrms = rox.rmsversion("1.2")
        msg = (
            "Not supported in this ROXAPI version. Points/polygons access "
            "to clipboard requires RMS {}".format(minimumrms)
        )
        raise NotImplementedError(msg)

    if stype == "horizon_picks":
        _roxapi_export_xyz_hpicks(
            xyzpp, rox, name, category, stype, realisation, attributes
        )

    if attributes:
        _roxapi_export_xyz_viafile(
            xyzpp,
            rox,
            name,
            category,
            stype,
            pfilter,
            realisation,
            attributes,
            is_polygons,
        )
    else:
        _roxapi_export_xyz(
            xyzpp, rox, name, category, stype, pfilter, realisation, is_polygons
        )


def _roxapi_export_xyz_hpicks(
    xyzpp, rox, name, category, stype, realisation, attributes
):  # pragma: no cover
    """
    Export/store as RMS horizon picks; this is only valid if points belong to wells
    """
    # need to think on design!
    raise NotImplementedError


def _roxapi_export_xyz_viafile(
    xyzpp, rox, name, category, stype, pfilter, realisation, attributes, is_polygons
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

    rox_xyz = _get_roxxyz(
        proj, name, category, stype, mode="set", is_polygons=is_polygons
    )

    # make a temporary folder and work within the with.. block
    with tempfile.TemporaryDirectory() as tmpdir:
        logger.info("Made a tmp folder: %s", tmpdir)
        ncount = xyzpp.to_file(
            os.path.join(tmpdir, "generic.rmsattr"),
            fformat="rms_attr",
            pfilter=pfilter,
            attributes=attributes,
        )

        if ncount:
            rox_xyz.load(
                os.path.join(tmpdir, "generic.rmsattr"), roxar.FileFormat.RMS_POINTS
            )


def _roxapi_export_xyz(
    self, rox, name, category, stype, pfilter, realisation, is_polygons
):  # pragma: no cover

    logger.warning("Realisation %s not in use", realisation)

    proj = rox.project
    if not _check_category_etc(proj, name, category, stype, realisation, mode="set"):
        raise RuntimeError("Cannot access correct category or name in RMS")

    rox_xyz = _get_roxxyz(
        proj, name, category, stype, mode="set", is_polygons=is_polygons
    )

    # pylint: disable=len-as-condition
    if self.dataframe is None or len(self.dataframe.index) == 0:
        return

    df = self.dataframe.copy()
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

    if is_polygons:
        arrxyz = []
        polys = df.groupby(self.pname)
        for _id, grp in polys:
            arr = np.stack([grp[self.xname], grp[self.yname], grp[self.zname]], axis=1)
            arrxyz.append(arr)
    else:
        xyz = df
        arrxyz = np.stack([xyz[self.xname], xyz[self.yname], xyz[self.zname]], axis=1)
    try:
        rox_xyz.set_values(arrxyz)
    except KeyError as kwe:
        logger.error(kwe)


def _check_category_etc(
    rox, name, category, stype, realisation, mode="get"
):  # pylint: disable=too-many-branches  # pragma: no cover

    """Helper to check if valid placeholder' whithin RMS."""

    logger.warning("Realisation %s not in use", realisation)

    stypedict = {
        "horizons": rox.project.horizons,
        "zones": rox.project.zones,
        "faults": rox.project.faults,
    }

    if stype in stypedict.keys():
        if name not in stypedict[stype]:
            logger.error("Cannot access name in stype=%s: %s", stype, name)
            return False
        if category not in stypedict[stype].representations:
            logger.error("Cannot access category in stype=%s: %s", stype, category)
            return False

    elif stype == "clipboard" and mode == "get":
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
                rox_xyz = rox.project.clipboard.folders[folders]
            except KeyError as keyerr:
                logger.error(
                    "Cannot access clipboards folder (not existing?): %s", keyerr
                )
                return False
        else:
            rox_xyz = rox.project.clipboard

        if name not in rox_xyz.keys():
            raise ValueError("Name {} is not within Clipboard...".format(name))

    elif stype == "clipboard" and mode == "set":
        logger.info("No need to check clipboard while setting data")
    else:
        raise ValueError("Invalid stype")

    return True


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
    elif stype == "clipboard":
        folders = None
        rox_xyz = rox.project.clipboard
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
                rox_xyz = rox.project.clipboard.folders[folders]

        if mode == "get":
            rox_xyz = rox_xyz[name]

        elif mode == "set":
            # clipboard folders will be created if not present, and overwritten else
            if is_polygons:
                rox_xyz = rox.project.clipboard.create_polylines(name, folders)
            else:
                rox_xyz = rox.project.clipboard.create_points(name, folders)

    else:
        raise TypeError("Unsupported stype: {}".format(stype))

    return rox_xyz


def _get_roxvalues(rox_xyz, realisation=0):  # pragma: no cover
    """Return the values from the Roxar API, either numpy (Points) or list(Polygons)."""
    try:
        roxitem = rox_xyz.get_values(realisation)
        logger.info(roxitem)
    except KeyError as kwe:
        logger.error(kwe)

    return roxitem


def _set_roxvalues(
    rox_xyz, values, is_polygons=False, realisation=0
):  # pragma: no cover
    """Set the values to Roxar API, either numpy (Points) or list(Polygons)."""
    pass
