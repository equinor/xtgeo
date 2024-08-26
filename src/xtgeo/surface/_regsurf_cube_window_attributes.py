"""Regular surface vs Cube, slice a window interval v4, in pure numpy.

This module is a refactored version of the original code, with the purpose of making
it faster and more reliable.

"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import interp1d

from xtgeo.common.log import null_logger

# from xtgeo.surface.regular_surface import surface_from_cube

if TYPE_CHECKING:
    from xtgeo.cube.cube1 import Cube

    from .regular_surface import RegularSurface


warnings.filterwarnings(action="ignore", message="All-NaN slice encountered")
warnings.filterwarnings(action="ignore", message="Mean of empty slice")
warnings.filterwarnings(action="ignore", message="Degree of freedom")

logger = null_logger(__name__)


STAT_ATTRS = [
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
SUM_ATTRS = [
    "sumpos",
    "sumneg",
    "sumabs",
]


@dataclass
class CubeAttrs:
    """Internal class for computing attributes in window between two surfaces."""

    this_surface: RegularSurface
    other_surface: RegularSurface
    cube: Cube
    ndiv: int = 10
    interpolation: str | int = "cubic"  # cf. scipy's interp1d 'kind' parameter
    minimum_thickness: float = 0.0

    # internal attributes
    _depth_cube: Cube = None
    _outside_depth: float = None  # will be detected from the depth cube
    _reduced_cube: Cube = None
    _reduced_depth_cube: Cube = None
    _refined_cube: Cube = None
    _refined_depth_cube: Cube = None

    _upper: RegularSurface = None  # upper surf, resampled to cube map resolution
    _lower: RegularSurface = None  # lower surf, resampled to cube map resolution
    _min_thickness_mask: RegularSurface = None  # mask for minimum thickness truncation

    _result_attr_maps: dict = field(default_factory=dict)  # holds the resulting maps

    def __post_init__(self):
        self._process_upper_lower_surface()
        self._create_depth_cube()
        self._create_reduced_cubes()
        self._refine_interpolate()
        self._depth_mask()
        self._compute_statistical_attribute_surfaces()

    def result(self) -> dict[RegularSurface]:
        # return the resulting attribute maps
        return self._result_attr_maps

    def _process_upper_lower_surface(self) -> None:
        """Extract upper and lower surface, sampled to cube resolution."""

        from xtgeo import surface_from_cube

        # determine which of "this" and "other" is actually upper and lower
        diff = self.other_surface - self.this_surface
        if diff.values.mean() > 0:
            upper, lower = self.this_surface, self.other_surface
        else:
            lower, upper = self.this_surface, self.other_surface

        # although not an attribute, we store the upper and lower surfaces
        self._result_attr_maps["upper"] = upper
        self._result_attr_maps["lower"] = lower

        # get the surfaces on cube resolution
        self._upper = surface_from_cube(self.cube, self.cube.zori)
        self._lower = surface_from_cube(self.cube, self.cube.zori)
        self._upper.resample(upper)
        self._lower.resample(lower)

        self._upper.fill()
        self._lower.fill()

        self._min_thickness_mask = self._lower - self._upper

        self._min_thickness_mask.values = np.where(
            self._min_thickness_mask.values <= self.minimum_thickness, 0, 1
        )
        if np.all(self._min_thickness_mask.values == 0):
            raise ValueError(
                "The minimum thickness is too large, no valid data in the interval. "
                "Perhaps surfaces are overlapping?"
            )

    def _create_depth_cube(self):
        """Create a cube where values are cube depths; to be used as filter.

        Belowe and above the input surfaces (plus a buffer), the values are set to
        a constant value self._outside_depth.
        """

        print("Create depth cube...")
        self._depth_cube = self.cube.copy()
        darr = [
            self.cube.zori + n * self.cube.zinc
            for n in range(self._depth_cube.values.shape[2])
        ]
        self._depth_cube.values[:, :, :] = darr

        # check that surfaces are within the cube
        if self._upper.values.min() > self._depth_cube.values.max():
            raise ValueError("Upper surface is fully below the cube")
        if self._lower.values.max() < self._depth_cube.values.min():
            raise ValueError("Lower surface is fully above the cube")
        if self._upper.values.max() < self._depth_cube.values.min():
            warnings.warn("Upper surface is fully above the cube", UserWarning)
        if self._lower.values.min() > self._depth_cube.values.max():
            warnings.warn("Lower surface is fully below the cube", UserWarning)

        self._outside_depth = self._depth_cube.values.max() + 1

        add_extra_depth = 2 * self.cube.zinc  # add buffer on upper/lower edges

        self._depth_cube.values = np.where(
            self._depth_cube.values < self._upper.values.min() - add_extra_depth,
            self._outside_depth,
            self._depth_cube.values,
        )

        self._depth_cube.values = np.where(
            self._depth_cube.values > self._lower.values.max() + add_extra_depth,
            self._outside_depth,
            self._depth_cube.values,
        )

    def _create_reduced_cubes(self) -> None:
        """Create a smaller cube based on the depth cube filter.

        The purpose is to limit the computation to the relevant volume, to save
        CPU time. I.e. cube values above the upper surface and below the lower are
        now excluded.
        """
        print("Create reduced cubes...")
        from xtgeo import Cube

        depthv = self._depth_cube.values

        cubev = self.cube.values.copy()  # copy, so we don't change the input instance
        cubev[self.cube.traceidcodes == 2] = np.nan  # set dead traces to nan

        # Create a boolean mask based on the threshold
        mask = depthv < self._outside_depth

        # Find the bounding box of the true values
        non_zero_indices = np.nonzero(mask)
        if len(non_zero_indices[0]) == 0:
            raise RuntimeError(  # should not occur?
                "No valid data found in the depth cube. Perhaps the surfaces are "
                "outside the cube?"
            )

        min_indices = np.min(non_zero_indices, axis=1)
        max_indices = (
            np.max(non_zero_indices, axis=1) + 1
        )  # Add 1 to include the upper bound

        # Extract the reduced cube using slicing
        red = cubev[
            min_indices[0] : max_indices[0],
            min_indices[1] : max_indices[1],
            min_indices[2] : max_indices[2],
        ]

        # Reduced depth cube
        red_depth = depthv[
            min_indices[0] : max_indices[0],
            min_indices[1] : max_indices[1],
            min_indices[2] : max_indices[2],
        ]

        zori = float(self._depth_cube.values.min())

        common_params = {
            "ncol": red.shape[0],
            "nrow": red.shape[1],
            "nlay": red.shape[2],
            "xinc": self.cube.xinc,
            "yinc": self.cube.yinc,
            "zinc": self.cube.zinc,
            "xori": self.cube.xori,
            "yori": self.cube.yori,
            "zori": zori,
            "rotation": self.cube.rotation,
            "yflip": self.cube.yflip,
        }

        self._reduced_cube = Cube(
            **common_params,
            values=red.astype(np.float32),
        )

        self._reduced_depth_cube = Cube(
            **common_params,
            values=red_depth.astype(np.float32),
        )

        logger.debug("Reduced cubes created %s", self._reduced_cube.values.shape)

    def _refine_interpolate(self):
        """Apply reduced cubes and interpolate to a finer grid vertically.

        This is done to get a more accurate representation of the cube values.
        """
        from xtgeo import Cube  # local import to avoid circular import

        logger.debug("Refine cubes and interpolate...")
        arr = self._reduced_cube.values
        arr_depth = self._reduced_depth_cube.values
        ndiv = self.ndiv

        # Create linear interpolation function along the last axis
        fdepth = interp1d(
            np.arange(arr_depth.shape[2]),
            arr_depth,
            axis=2,
            kind="linear",
            assume_sorted=True,
        )

        # Create interpolation function along the last axis
        fcube = interp1d(
            np.arange(arr.shape[2]),
            arr,
            axis=2,
            kind=self.interpolation,
            assume_sorted=True,
        )

        # Define new sampling points along the last axis
        new_z = np.linspace(0, arr.shape[2] - 1, arr.shape[2] * ndiv)

        # Resample the cube array
        resampled_arr = fcube(new_z)

        # Resample the depth array (always linear)
        resampled_depth_arr = fdepth(new_z)

        new_zinc = (resampled_depth_arr.max() - resampled_depth_arr.min()) / (
            resampled_depth_arr.shape[2] - 1
        )
        common_params = {
            "ncol": resampled_arr.shape[0],
            "nrow": resampled_arr.shape[1],
            "nlay": resampled_arr.shape[2],
            "xinc": self.cube.xinc,
            "yinc": self.cube.yinc,
            "zinc": new_zinc,
            "xori": self.cube.xori,
            "yori": self.cube.yori,
            "zori": resampled_depth_arr.min(),
            "rotation": self._reduced_cube.rotation,
            "yflip": self._reduced_cube.yflip,
        }

        self._refined_cube = Cube(
            **common_params,
            values=resampled_arr.astype(np.float32),
        )
        self._refined_depth_cube = Cube(
            **common_params,
            values=resampled_depth_arr.astype(np.float32),
        )

    def _depth_mask(self):
        """Set nan values outside the interval defined by the upper + lower surface.

        In addition, set nan values where the thickness is less than the minimum.

        """
        print("Depth mask...")

        depth_2d_upper_exp = np.expand_dims(self._upper.values, 2)
        depth_2d_lower_exp = np.expand_dims(self._lower.values, 2)
        mask_2d_exp = np.expand_dims(self._min_thickness_mask.values, 2)

        depth_3d = self._refined_depth_cube.values

        self._refined_cube.values = np.where(
            (depth_3d < depth_2d_upper_exp)
            | (depth_3d > depth_2d_lower_exp)
            | (mask_2d_exp == 0),
            np.nan,
            self._refined_cube.values,
        ).astype(np.float32)

        # similar for reduced cubes with original resolution
        depth_3d = self._reduced_depth_cube.values

        self._reduced_cube.values = np.where(
            (depth_3d < depth_2d_upper_exp)
            | (depth_3d > depth_2d_lower_exp)
            | (mask_2d_exp == 0),
            np.nan,
            self._reduced_cube.values,
        ).astype(np.float32)

    def _compute_statistical_attribute_surfaces(self) -> None:
        """Compute the attributes and update the dict that holds all such maps"""

        print("Compute statistical attributes...")

        def _add_to_attribute_map(attr_name, values) -> None:
            """Compute the attribute map and add to result dictionary."""
            attr_map = self._upper.copy()
            attr_map.values = np.ma.masked_invalid(values)

            # now resample to the original input map
            attr_map_resampled = self.this_surface.copy()
            attr_map_resampled.resample(attr_map)

            attr_map_resampled.values = np.ma.masked_where(
                self.this_surface.values.mask, attr_map_resampled.values
            )

            self._result_attr_maps[attr_name] = attr_map_resampled

        def _safe_nanvar(arr_slice):
            """Variance, incl. handling when number of non-NaN elements is <= 1."""
            if np.sum(~np.isnan(arr_slice)) > 1:
                return np.nanvar(arr_slice)
            return np.nan

        cref = self._refined_cube.values
        with np.errstate(invalid="ignore"):
            for attr in STAT_ATTRS:
                if attr == "mean":
                    values = np.nanmean(cref, axis=2)
                elif attr == "var":
                    values = np.apply_along_axis(_safe_nanvar, axis=2, arr=cref)
                elif attr == "max":
                    values = np.nanmax(cref, axis=2)
                elif attr == "min":
                    values = np.nanmin(cref, axis=2)
                elif attr == "rms":
                    values = np.sqrt(np.nanmean(np.square(cref), axis=2))
                elif attr == "maxneg":
                    values = np.nanmin(np.where(cref < 0, cref, np.inf), axis=2)
                    values[values == np.inf] = np.nan
                elif attr == "maxpos":
                    values = np.nanmax(np.where(cref >= 0, cref, np.nan), axis=2)
                elif attr == "maxabs":
                    values = np.nanmax(np.abs(cref), axis=2)
                elif attr == "meanabs":
                    values = np.nanmean(np.abs(cref), axis=2)
                elif attr == "meanpos":
                    # avoid RuntimeWarning: Mean of empty slice
                    if np.any(np.nansum(np.where(cref > 0, cref, 0), axis=2) > 0):
                        values = np.nanmean(np.where(cref >= 0, cref, np.nan), axis=2)
                    else:
                        values = np.full(cref.shape[:2], np.nan)
                elif attr == "meanneg":
                    # avoid RuntimeWarning: Mean of empty slice
                    if np.any(np.nansum(np.where(cref < 0, cref, 0), axis=2) < 0):
                        values = np.nanmean(np.where(cref < 0, cref, np.nan), axis=2)
                    else:
                        values = np.full(cref.shape[:2], np.nan)
                else:
                    continue

                _add_to_attribute_map(attr, values)

        # Compute sum attributes, but on the original cube resolution (counting!)
        cref = self._reduced_cube.values
        for attr in SUM_ATTRS:
            if attr == "sumneg":
                values = np.nansum(np.where(cref < 0, cref, np.nan), axis=2)
                values = np.ma.masked_greater_equal(
                    values, 0.0
                )  # to make undefined map areas
            elif attr == "sumpos":
                values = np.nansum(np.where(cref >= 0, cref, np.nan), axis=2)
                values = np.ma.masked_less_equal(
                    values, 0.0
                )  # to make undefined map areas
            elif attr == "sumabs":
                values = np.nansum(np.abs(cref), axis=2)
            else:
                raise ValueError(f"The attribute name {attr} is not supported")

            _add_to_attribute_map(attr, values)
