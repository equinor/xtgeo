"""Regular surface vs Cube, slice a window interval v3, in pure numpy."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.interpolate import interp1d

import xtgeo
from xtgeo.common import null_logger

warnings.filterwarnings(action="ignore", message="All-NaN slice encountered")
warnings.filterwarnings(action="ignore", message="Mean of empty slice")
warnings.filterwarnings(action="ignore", message="Degree of freedom")

logger = null_logger(__name__)


STATATTRS = [
    "min",
    "max",
    "mean",
    "var",
    "rms",
    "maxpos",
    "maxneg",
    "maxabs",
    "meanabs",
    "meanpos",
    "meanneg",
]
SUMATTRS = [
    "sumpos",
    "sumneg",
    "sumabs",
]
ALLATTRS = STATATTRS + SUMATTRS

# self ~ RegularSurface() instance


def _cut_cube_deadtraces(cube: xtgeo.Cube, deadtraces: bool) -> np.ndarray:
    """Take the cube numpy values and filter away dead traces as np.nan."""
    logger.info("Assign dead traces")
    dvalues = cube.values.copy()

    if deadtraces and 2 in cube.traceidcodes:
        dvalues[cube.traceidcodes == 2] = np.nan
        logger.info("Dead traces encountered in this cube, set values to np.nan")
    else:
        logger.info("No dead traces encountered in this cube, or deadtraces is False")

    return dvalues


def _get_iso_maskthreshold_surface(
    upper: xtgeo.RegularSurface,
    lower: xtgeo.RegularSurface,
    maskthreshold: float,
) -> xtgeo.RegularSurface:
    """Return a surface with value 0 where isochore <= threshold"""
    logger.info("Maskthreshold based on isochore")
    result = upper.copy()
    result.fill()
    diff = lower - upper
    result.values = np.ma.where(diff.values <= maskthreshold, 0, 1)
    return result


def _proxy_surf_outside_cube(
    self,
    cube: xtgeo.Cube,
) -> xtgeo.RegularSurface:
    """Proxy for the part of input surface that is outside the cube area."""
    logger.info("Get a proxy for part of original surface being outside the cube")
    outside = self.copy()
    outside.values = 0.0

    tmp_surf = xtgeo.surface_from_cube(cube, 0.0)
    boundary = tmp_surf.get_boundary_polygons()
    outside.set_outside(boundary, 1.0)
    return outside  # is 1 outside the cube area, 0 within the cube


def _upper_lower_surface(
    self,
    cube: xtgeo.Cube,
    zsurf: xtgeo.RegularSurface,
    other: xtgeo.RegularSurface,
    other_position: str,
    zrange: float,
) -> tuple[xtgeo.RegularSurface, xtgeo.RegularSurface]:
    """Return upper and lower surface, sampled to cube resolution."""

    logger.info("Define surfaces to apply...")
    this = zsurf if zsurf is not None else self.copy()

    if other is not None:
        if other_position.lower() == "below":
            surf1 = this
            surf2 = other
        else:
            surf1 = other  # avoid changing self instance
            surf2 = this
    else:
        surf1 = this.copy()
        surf2 = this.copy()
        surf1.values -= zrange
        surf2.values += zrange

    # get the surfaces on cube resolution
    upper = xtgeo.surface_from_cube(cube, 0)
    lower = upper.copy()
    upper.resample(surf1)
    lower.resample(surf2)

    logger.info(
        "Return resmapled surfaces, avg for upper and lower is %s, %s",
        upper.values.mean(),
        lower.values.mean(),
    )
    return upper, lower


def _create_depth_cube(cube: xtgeo.Cube) -> np.ndarray:
    """Create a cube (np array) where values are cube depths; to be used as filter."""
    logger.info("Create a depth cube...")
    dcube = cube.values.copy()
    darr = [cube.zori + n * cube.zinc for n in range(dcube.shape[2])]
    dcube[:, :, :] = darr

    logger.info("Created a depth cube starting from %s", np.mean(dcube))

    return dcube


def _refine_cubes_vertically(cvalues, dvalues, ndiv):
    """Refine the cubes vertically for better resolution"""
    if not ndiv:
        ndiv = 2  # default
    logger.info("Resampling vertically, according to ndiv = %s", ndiv)
    if ndiv <= 1:
        logger.info("ndiv is less or equal to 1; no refinement done")
        return cvalues, dvalues

    logger.info("Original shape is %s", cvalues.shape)
    cref = np.random.rand(cvalues.shape[0], cvalues.shape[1], ndiv * cvalues.shape[2])
    dref = cref.copy()

    num_points = cref.shape[-1]
    # Create interpolation function for cube values
    interp_func1 = interp1d(
        np.arange(cvalues.shape[-1]),
        cvalues,
        axis=-1,
        kind="linear",
        fill_value="extrapolate",
    )
    interp_func2 = interp1d(
        np.arange(dvalues.shape[-1]),
        dvalues,
        axis=-1,
        kind="linear",
        fill_value="extrapolate",
    )
    # Resample array2 to match the number of points in array1
    cref = interp_func1(np.linspace(0, cvalues.shape[-1] - 1, num_points))
    dref = interp_func2(np.linspace(0, dvalues.shape[-1] - 1, num_points))

    logger.info("Resampling done, new shape is %s", cref.shape)
    return cref, dref


def _filter_cube_values_upper_lower(cvalues, dvalues, upper, lower):
    """Filter the cube (cvalues) based on depth interval."""

    nnans = np.count_nonzero(np.isnan(cvalues))
    logger.info("Filter cube in depth... number of nans is %s", nnans)

    upv = np.expand_dims(upper.values, axis=2)
    lov = np.expand_dims(lower.values, axis=2)

    cvalues[dvalues < upv] = np.nan
    cvalues[dvalues > lov] = np.nan

    nnans = np.count_nonzero(np.isnan(cvalues))
    ndefi = np.count_nonzero(~np.isnan(cvalues))
    logger.info("Filter cube in depth done, updated number of nans is %s", nnans)
    logger.info("Filter cube in depth done, remaining is %s", ndefi)
    return cvalues


def _expand_attributes(attribute: str | list) -> list:
    """The 'attribute' may be a name, 'all', or a list of attributes"""
    useattrs = None
    if isinstance(attribute, str):
        if attribute == "all":
            useattrs = ALLATTRS
        else:
            useattrs = [attribute]
    else:
        useattrs = attribute

    if not all(item in ALLATTRS for item in useattrs):
        raise ValueError(
            f"One or more values are not a valid, input list is {useattrs}, "
            f"allowed list is {ALLATTRS}"
        )
    return useattrs


def _compute_stats(
    cref: np.ndarray,
    attr: str,
    self: xtgeo.RegularSurface,
    upper: xtgeo.RegularSurface,
    masksurf: xtgeo.RegularSurface,
    sampling: str,
    snapxy: bool,
) -> xtgeo.RegularSurface:
    """Compute the attribute and return the attribute map"""
    logger.info("Compute stats...")

    if attr == "mean":
        values = np.nanmean(cref, axis=2)
    elif attr == "var":
        values = np.nanvar(cref, axis=2)
    elif attr == "max":
        values = np.nanmax(cref, axis=2)
    elif attr == "min":
        values = np.nanmin(cref, axis=2)
    elif attr == "rms":
        values = np.sqrt(np.nanmean(np.square(cref), axis=2))
    elif attr == "maxneg":
        use = cref.copy()
        use[cref >= 0] = np.nan
        values = np.nanmin(use, axis=2)
    elif attr == "maxpos":
        use = cref.copy()
        use[cref < 0] = np.nan
        values = np.nanmax(use, axis=2)
    elif attr == "maxabs":
        use = cref.copy()
        use = np.abs(use)
        values = np.nanmax(use, axis=2)
    elif attr == "meanabs":
        use = cref.copy()
        use = np.abs(use)
        values = np.nanmean(use, axis=2)
    elif attr == "meanpos":
        use = cref.copy()
        use[cref < 0] = np.nan
        values = np.nanmean(use, axis=2)
    elif attr == "meanneg":
        use = cref.copy()
        use[cref >= 0] = np.nan
        values = np.nanmean(use, axis=2)
    elif attr == "sumneg":
        use = cref.copy()
        use[cref >= 0] = np.nan
        values = np.nansum(use, axis=2)
        values = np.ma.masked_greater_equal(values, 0.0)  # to make undefined map areas
    elif attr == "sumpos":
        use = cref.copy()
        use[cref < 0] = np.nan
        values = np.nansum(use, axis=2)
        values = np.ma.masked_less_equal(values, 0.0)  # to make undefined map areas
    elif attr == "sumabs":
        use = cref.copy()
        use = np.abs(use)
        values = np.nansum(use, axis=2)
    else:
        raise ValueError(f"The attribute name {attr} is not supported")

    actual = self.copy()
    sampled = upper.copy()

    sampled.values = np.ma.masked_invalid(values)
    sampled.values = np.ma.masked_where(masksurf.values == 0, sampled.values)

    if not snapxy:
        sampling_option = "nearest" if sampling in ("nearest", "cube") else "bilinear"
        actual.resample(
            sampled, sampling=sampling_option
        )  # will be on input map resolution
    else:
        actual = sampled

    logger.info("Compute stats... done")
    return actual


def slice_cube_window(
    self,
    cube: xtgeo.Cube,
    zsurf: xtgeo.RegularSurface | None = None,
    other: xtgeo.RegularSurface | None = None,
    other_position: str = "below",
    sampling: str = "nearest",
    mask: bool = True,
    zrange: float = 10.0,
    ndiv: int | None = None,
    attribute: str = "max",
    maskthreshold: float = 0.1,
    snapxy: bool = False,
    showprogress: bool = False,
    deadtraces: bool = True,
):
    """Main entry point towards caller"""
    if showprogress:
        print("progress: initialising for attributes...")

    cvalues = _cut_cube_deadtraces(cube, deadtraces)

    upper, lower = _upper_lower_surface(
        self, cube, zsurf, other, other_position, zrange
    )

    outside_proxy = None
    if not mask:
        outside_proxy = _proxy_surf_outside_cube(self, cube)

    masksurf = _get_iso_maskthreshold_surface(upper, lower, maskthreshold)

    dvalues = _create_depth_cube(cube)

    apply_ndiv = 1 if sampling == "cube" else ndiv
    if showprogress:
        print(f"progress: refine according to actual ndiv = {apply_ndiv}...")

    cref, dref = _refine_cubes_vertically(cvalues, dvalues, apply_ndiv)

    cref = _filter_cube_values_upper_lower(cref, dref, upper, lower)

    cval = _filter_cube_values_upper_lower(cvalues, dvalues, upper, lower)

    use_attrs = _expand_attributes(attribute)

    attrs = {}
    if showprogress:
        print("progress: compute mean, variance, etc attributes...")
    for attr in use_attrs:
        if attr in SUMATTRS:
            # use cval, which is not refined vertically
            res = _compute_stats(cval, attr, self, upper, masksurf, sampling, snapxy)
        else:
            res = _compute_stats(cref, attr, self, upper, masksurf, sampling, snapxy)

        if outside_proxy and not snapxy:
            res.values = np.ma.where(outside_proxy > 0, self.values, res.values)

        attrs[attr] = res

    # if attribute is str, self shall be updated and None returned,
    # otherwise a dict of attributes objects shall be returned
    if isinstance(attrs, dict) and len(attrs) == 1 and isinstance(attribute, str):
        self.values = attrs[attribute].values
        return None

    return attrs
