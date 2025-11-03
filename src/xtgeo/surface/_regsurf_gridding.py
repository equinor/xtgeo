"""Do gridding from 3D parameters"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal

import gstools as gs
import numpy as np
import numpy.ma as ma
import scipy.interpolate
import scipy.ndimage
from numpy.typing import NDArray
from scipy.spatial import cKDTree

import xtgeo._internal as _internal
from xtgeo.common.constants import UNDEF, UNDEF_LIMIT
from xtgeo.common.log import null_logger
from xtgeo.xyz.polygons import Polygons

if TYPE_CHECKING:
    from collections.abc import Callable

    from xtgeo.surface import RegularSurface
    from xtgeo.xyz import Points

logger = null_logger(__name__)

# Type aliases
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int32]
BoolArray = NDArray[np.bool_]
MaskedArray = ma.MaskedArray[Any, np.dtype[np.float64]]

# Note: 'self' is an instance of RegularSurface

# ======================================================================================
# HELPER FUNCTIONS (internal for this module)
# ======================================================================================


def merge_close_points_preprocessing(
    points: Points,
    merge_close_points: float | None = None,
    merge_method: Literal["average", "median", "first", "min_z", "max_z"] = "average",
) -> Points:
    """
    Preprocess points by merging those that are too close together.

    This is a common preprocessing step for all gridding methods to avoid numerical
    issues when points are very close to each other.

    Args:
        points: Points object to potentially merge
        merge_close_points: Minimum distance threshold (in map units) for merging
            close points. Set to None to disable merging.
        merge_method: Method for merging close points ('average', 'median', 'first',
            'min_z', 'max_z'). Default is 'average'.

    Returns:
        Points object (either original or a merged copy)
    """
    if merge_close_points is None:
        return points

    threshold = float(merge_close_points)

    # Make a copy before merging to avoid modifying original
    working_points = points.copy()
    working_points.merge_close_points(min_distance=threshold, method=merge_method)

    logger.debug(
        "Merged close points: %d -> %d (threshold=%.2f, method=%s)",
        points.nrow,
        working_points.nrow,
        threshold,
        merge_method,
    )

    return working_points


def _check_close_points_warning(
    surface: RegularSurface, points: Points, threshold_factor: float = 0.1
) -> None:
    """
    Check if points are too close together and emit a warning.

    This is useful for methods like radial basis and kriging where very close points
    can cause numerical instability or ill-conditioned matrices.

    Args:
        surface: RegularSurface instance (for calculating avg_inc)
        points: Points object to check
        threshold_factor: Fraction of avg_inc to use as threshold (default 0.1)

    Returns:
        None (emits warning if close points detected)
    """
    if points.nrow < 2:
        return

    avg_inc = (surface.xinc + surface.yinc) / 2.0
    threshold = threshold_factor * avg_inc

    dfra = points.get_dataframe(copy=False)
    coords = dfra[[points.xname, points.yname]].values

    # Use KDTree to find nearest neighbors efficiently
    tree = cKDTree(coords)
    # Query for 2 nearest neighbors (first will be the point itself)
    distances, _ = tree.query(coords, k=2)
    # Get minimum distance to nearest neighbor (excluding self)
    min_distance = distances[:, 1].min()

    if min_distance < threshold:
        import warnings

        warnings.warn(
            f"Points are very close together (minimum distance: {min_distance:.2f} "
            f"< {threshold_factor}*avg_inc = {threshold:.2f}). "
            f"This may cause numerical issues with this gridding method. "
            f"Consider using Points.merge_close_points() to merge nearby points "
            f"before gridding, e.g. with min_distance = 10% of the mapping increment",
            RuntimeWarning,
            stacklevel=4,
        )
        logger.warning(
            "Close points detected: min_distance=%.2f, threshold=%.2f",
            min_distance,
            threshold,
        )


def _points_gridding_common(
    self: RegularSurface, input_points: Points, coarsen: int
) -> tuple[Any, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """
    Do gridding from a points data set - common preprocessing.

    Args:
        self: RegularSurface instance
        input_points: Points object (already merged if needed)
        coarsen: Coarsening factor for points

    Returns:
        tuple: (dfra, xcv, ycv, zcv, xiv, yiv) where xcv, ycv, zcv are filtered
    """

    points = _filter_points_within_surface(self, input_points)

    xiv, yiv = self.get_xy_values()

    dfra = points.get_dataframe(copy=False)

    xcv = dfra[points.xname].values
    ycv = dfra[points.yname].values
    zcv = dfra[points.zname].values

    if coarsen > 1:
        xcv = xcv[::coarsen]
        ycv = ycv[::coarsen]
        zcv = zcv[::coarsen]

    return dfra, xcv, ycv, zcv, xiv, yiv


def _filter_points_within_surface(self: RegularSurface, points: Points) -> Points:
    """
    Filter points to only those within surface boundaries.

    Args:
        self: RegularSurface instance
        points: Input xtgeo.Points

    Returns:
        An updated xtgeo.Points instance
    """
    p = points.copy()

    logger.debug("Number of points before filtering: %d", p.nrow)

    r = _internal.regsurf.RegularSurface(self).get_outer_corners()
    plist = [
        (r[0].x, r[0].y, r[0].z, 0),
        (r[1].x, r[1].y, r[1].z, 0),
        (r[3].x, r[3].y, r[3].z, 0),
        (r[2].x, r[2].y, r[2].z, 0),
        (r[0].x, r[0].y, r[0].z, 0),
    ]
    boundary = Polygons(plist)

    p.eli_outside_polygons(boundary)

    logger.debug("Filtered points inside surface: %d", p.nrow)

    return p


def _calculate_optimal_epsilon(
    xcv: FloatArray, ycv: FloatArray, k_neighbors: int = 5
) -> float:
    """
    Calculate optimal epsilon parameter for RBF kernels based on point spacing.

    Uses the mean distance to k nearest neighbors to be more robust to clustering.

    Args:
        xcv: X coordinates of points
        ycv: Y coordinates of points
        k_neighbors: Number of neighbors to consider (default 5)

    Returns:
        float: Optimal epsilon value
    """

    # Sample points if too many (for performance)
    n_sample = min(1000, len(xcv))
    if len(xcv) > n_sample:
        idx = np.random.choice(len(xcv), n_sample, replace=False)
        sample_x = xcv[idx]
        sample_y = ycv[idx]
    else:
        sample_x = xcv
        sample_y = ycv

    sample_coords = np.column_stack([sample_x, sample_y])

    tree = cKDTree(sample_coords)

    # Find distance to k nearest neighbors (k+1 because first is the point itself)
    k_neighbors = min(k_neighbors + 1, len(sample_coords))
    distances, _ = tree.query(sample_coords, k=k_neighbors)

    # Take mean of the k nearest neighbors
    mean_k_distances = distances[:, 1:].mean(axis=1)

    epsilon = np.median(mean_k_distances)

    # Safety check: ensure minimum based on data extent
    x_range = sample_x.max() - sample_x.min()
    y_range = sample_y.max() - sample_y.min()
    extent = np.sqrt(x_range**2 + y_range**2)
    min_epsilon = extent / n_sample

    epsilon = max(epsilon, min_epsilon)

    logger.debug(
        "Auto-calculated epsilon: %.2f "
        "(based on median of mean %d-neighbor distances from %d samples)",
        epsilon,
        k_neighbors - 1,
        n_sample,
    )

    warnings.warn(
        f"Epsilon auto-calculated as {epsilon:.2f}. For better control and "
        "to avoid artifacts from duplicate/close points, consider merging nearby "
        "points using Points.merge_close_points() before gridding.",
        UserWarning,
        stacklevel=3,
    )

    return epsilon


def _calculate_optimal_radius(
    self: RegularSurface, xcv: FloatArray, ycv: FloatArray
) -> float:
    """
    Calculate optimal search radius based on grid resolution and point density.

    Args:
        self: RegularSurface instance
        xcv: X coordinates of points
        ycv: Y coordinates of points

    Returns:
        float: Optimal search radius in map units
    """
    # Base radius on grid increments (cover adjacent cells)
    grid_based_radius = 2.0 * max(self.xinc, self.yinc)

    # Estimate point density and spacing
    x_range = xcv.max() - xcv.min()
    y_range = ycv.max() - ycv.min()
    area = x_range * y_range

    if area > 0 and len(xcv) > 0:
        point_density = len(xcv) / area
        avg_point_distance = 1.0 / np.sqrt(point_density)

        # Use the larger of grid-based or point-based radius
        # This ensures we capture enough points while respecting grid resolution
        radius = max(grid_based_radius, 1.5 * avg_point_distance)

        logger.debug(
            "Auto-calculated search radius: %.2f "
            "(grid-based: %.2f, point-based: %.2f, xinc=%.2f, yinc=%.2f)",
            radius,
            grid_based_radius,
            1.5 * avg_point_distance,
            self.xinc,
            self.yinc,
        )
    else:
        radius = grid_based_radius
        logger.debug(
            "Auto-calculated search radius: %.2f (based on grid increments only)",
            radius,
        )

    return radius


def _estimate_correlation_length(
    xcv: FloatArray, ycv: FloatArray, zcv: FloatArray
) -> float:
    """
    Estimate correlation length (range_value) from point data.

    Uses the empirical variogram to estimate where correlation drops to ~5% of sill.

    Args:
        xcv: X coordinates
        ycv: Y coordinates
        zcv: Z values

    Returns:
        float: Estimated correlation length
    """

    # Sample points for efficiency
    n_sample = min(500, len(xcv))
    if len(xcv) > n_sample:
        idx = np.random.choice(len(xcv), n_sample, replace=False)
        sample_x = xcv[idx]
        sample_y = ycv[idx]
        sample_z = zcv[idx]
    else:
        sample_x = xcv
        sample_y = ycv
        sample_z = zcv

    coords = np.column_stack([sample_x, sample_y])
    tree = cKDTree(coords)

    # Calculate semi-variogram for different distance bins
    max_dist = (
        np.sqrt(
            (sample_x.max() - sample_x.min()) ** 2
            + (sample_y.max() - sample_y.min()) ** 2
        )
        / 3
    )

    n_bins = 20
    bin_edges = np.linspace(0, max_dist, n_bins + 1)
    gamma = []
    bin_centers = []

    for i in range(n_bins):
        d_min = bin_edges[i]
        d_max = bin_edges[i + 1]

        # Find pairs in this distance range
        pairs = []
        for j, coord in enumerate(coords):
            idx_neighbors = tree.query_ball_point(coord, d_max)
            for k in idx_neighbors:
                if k > j:  # Avoid duplicates
                    dist = np.linalg.norm(coords[j] - coords[k])
                    if d_min <= dist < d_max:
                        pairs.append((j, k))

        if len(pairs) > 10:  # Need enough pairs
            pairs_z = [(sample_z[j] - sample_z[k]) ** 2 for j, k in pairs]
            gamma.append(0.5 * np.mean(pairs_z))
            bin_centers.append((d_min + d_max) / 2)

    if len(gamma) > 3:
        # ~ range_value as distance where variogram reaches ~63% of sill (1 - 1/e)
        sill = max(gamma)
        threshold = 0.63 * sill

        for dist, g in zip(bin_centers, gamma):
            if g >= threshold:
                return dist

        # If not found, use 1/3 of max distance
        return max_dist

    # Fallback: use mean nearest neighbor distance * 3
    distances, _ = tree.query(coords, k=2)
    return 3.0 * np.mean(distances[:, 1])


# ======================================================================================
# FUNCTIONS doing gridding - to be called by public module
# ======================================================================================


def points_simple_gridding(
    self: RegularSurface,
    points: Points,
    method: Literal["linear", "nearest", "cubic"] = "linear",
    coarsen: int = 1,
) -> None:
    """
    Triangulation-based gridding using scipy.interpolate.griddata.

    Fast and reliable for dense point datasets.
    Methods: 'linear', 'nearest', 'cubic'

    Args:
        points: Points object (may already be merged)
        method: Interpolation method
        coarsen: Coarsening factor
    """
    dfra, xcv, ycv, zcv, xiv, yiv = _points_gridding_common(
        self,
        points,
        coarsen=coarsen,
    )

    validmethods = ["linear", "nearest", "cubic"]
    if method not in set(validmethods):
        raise ValueError(
            f"Invalid method for gridding: {method}, valid options are {validmethods}"
        )

    npoints = len(xcv)
    logger.debug("Gridding %d points with method '%s'...", npoints, method)

    try:
        znew = scipy.interpolate.griddata(
            (xcv, ycv), zcv, (xiv, yiv), method=method, fill_value=np.nan
        )

    except ValueError as verr:
        raise RuntimeError(f"Could not do gridding: {verr}")

    logger.debug("Gridding point ... DONE")

    self._ensure_correct_values(znew)


def points_rbf_gridding(
    self: RegularSurface,
    points: Points,
    function: str = "thin_plate_spline",
    smoothing: float = 0.0,
    epsilon: float | None = None,
    coarsen: int = 1,
) -> None:
    """
    Do Radial Basis Function (RBF) gridding from a points data set.

    RBF is excellent for sparse data and provides smooth interpolation
    similar to 'cubic' but with better control over smoothness. It will also work nicely
    with more dense data, in particular when using a linear function (kernel).

    Args:
        points: Points object containing x, y, z coordinates
        function: RBF kernel type. Options:
            - 'thin_plate_spline': Thin plate spline (recommended, similar to cubic)
            - 'cubic': Cubic RBF (smoother than thin_plate_spline)
            - 'quintic': Quintic RBF (very smooth)
            - 'linear': Linear RBF, good for dense data (less smooth)
            - 'multiquadric': Multiquadric (requires epsilon)
            - 'inverse_multiquadric': Inverse multiquadric (requires epsilon)
            - 'inverse_quadratic': Inverse quadratic (requires epsilon)
            - 'gaussian': Gaussian (requires epsilon)
        smoothing: Smoothing parameter (0 = interpolating, >0 = smoothing).
            For noisy data, try 0.1 to 1.0
        epsilon: Shape parameter for certain kernels. If None and required by kernel,
            automatically calculated as the mean distance between points.
        coarsen: Coarsening factor for input points
    """
    dfra, xcv, ycv, zcv, xiv, yiv = _points_gridding_common(
        self, points, coarsen=coarsen
    )

    # Check for close points that may cause numerical issues
    _check_close_points_warning(self, points, threshold_factor=0.1)

    npoints = len(xcv)

    # Kernels that require epsilon parameter
    epsilon_required = {
        "multiquadric",
        "inverse_multiquadric",
        "inverse_quadratic",
        "gaussian",
    }

    # Auto-calculate epsilon if needed but not provided
    if function in epsilon_required and epsilon is None:
        epsilon = _calculate_optimal_epsilon(xcv, ycv)

    logger.debug(
        "RBF gridding from %d points (function=%s, smooth=%s, epsilon=%s)...",
        npoints,
        function,
        smoothing,
        epsilon,
    )

    try:
        rbf = scipy.interpolate.RBFInterpolator(
            np.column_stack([xcv, ycv]),
            zcv,
            kernel=function,
            smoothing=smoothing,
            epsilon=epsilon,
        )
        znew = rbf(np.column_stack([xiv.ravel(), yiv.ravel()])).reshape(xiv.shape)

    except (ValueError, TypeError) as err:
        raise RuntimeError(f"Could not do RBF gridding: {err}")

    logger.debug("RBF gridding ... DONE")
    self._ensure_correct_values(znew)


def points_moving_average_gridding(
    self: RegularSurface,
    points: Points,
    radius: float | None = None,
    min_points: int = 3,
    coarsen: int = 1,
) -> None:
    """
    Do moving average gridding from a points data set.

    Simple averaging of all points within a search radius.

    Args:
        points: Points object containing x, y, z coordinates
        radius: Search radius in map units. If None, automatically calculated
                based on grid resolution and point density.
        min_points: Minimum number of points required for averaging, default 3
        coarsen: Coarsening factor for input points
    """
    dfra, xcv, ycv, zcv, xiv, yiv = _points_gridding_common(
        self, points, coarsen=coarsen
    )

    # Calculate optimal radius if not provided
    if radius is None:
        radius = _calculate_optimal_radius(self, xcv, ycv)
        logger.debug("Applied radius: %s", radius)

    logger.debug(
        "Moving average gridding from %d points (radius=%.1f, min_points=%d)...",
        len(xcv),
        radius,
        min_points,
    )

    # Use scipy's griddata with nearest neighbor to get initial grid
    # Then apply moving average
    try:
        # First, grid the data with nearest neighbor to get point values on grid
        znear = scipy.interpolate.griddata(
            (xcv, ycv), zcv, (xiv, yiv), method="nearest", fill_value=np.nan
        )

        # Note: No need to grid a count map here; we derive counts via convolution below

        # Now apply a true moving average using a disk-shaped kernel
        # Convert radius to grid cells
        radius_cells_x = radius / self.xinc
        radius_cells_y = radius / self.yinc

        # Create a circular/elliptical structuring element
        y_kernel = int(np.ceil(radius_cells_y))
        x_kernel = int(np.ceil(radius_cells_x))

        y_grid, x_grid = np.ogrid[-y_kernel : y_kernel + 1, -x_kernel : x_kernel + 1]

        # Elliptical distance for rotated/anisotropic grids
        kernel = ((x_grid / radius_cells_x) ** 2 + (y_grid / radius_cells_y) ** 2) <= 1

        # Apply uniform filter for averaging
        # Handle NaN values by tracking valid cells
        valid_mask = ~np.isnan(znear)

        # Sum of values
        znear_filled = np.where(valid_mask, znear, 0.0)
        sum_grid = scipy.ndimage.convolve(
            znear_filled, kernel.astype(float), mode="constant", cval=0.0
        )

        # Count of valid cells in each window
        count = scipy.ndimage.convolve(
            valid_mask.astype(float), kernel.astype(float), mode="constant", cval=0.0
        )

        # Calculate average, masking where not enough points
        znew = np.where(count >= min_points, sum_grid / count, np.nan)

    except (ValueError, TypeError) as err:
        raise RuntimeError(f"Could not do moving average gridding: {err}")

    logger.debug("Moving average gridding ... DONE")

    self._ensure_correct_values(znew)


def points_idw_gridding(
    self: RegularSurface,
    points: Points,
    power: float = 2.0,
    radius: float | None = None,
    min_points: int = 1,
    coarsen: int = 1,
) -> None:
    """
    Do inverse distance weighted (IDW) gridding from a points data set.

    Args:
        points: Points object containing x, y, z coordinates
        power: Power parameter for IDW (default 2.0). Higher values give more
               weight to nearby points
        radius: Search radius in map units. If None, uses all points for small
               datasets (< 1000 points) or auto-calculates radius for larger datasets
        min_points: Minimum number of points required for interpolation
        coarsen: Coarsening factor for input points (use every Nth point)
    """

    _, xcv, ycv, zcv, xiv, yiv = _points_gridding_common(self, points, coarsen=coarsen)

    npoints = len(xcv)

    # For large datasets without specified radius, auto-calculate one
    # to avoid O(n*m) complexity
    use_kdtree = radius is not None or npoints > 1000

    if use_kdtree and radius is None:
        radius = _calculate_optimal_radius(self, xcv, ycv)
        logger.debug(
            "Auto-calculated IDW search radius for large dataset: %.2f", radius
        )

    # Flatten grid coordinates for processing
    xi_flat = xiv.ravel()
    yi_flat = yiv.ravel()
    zi_flat = np.full(xi_flat.shape, np.nan)

    if use_kdtree:
        tree = cKDTree(np.column_stack([xcv, ycv]))
        logger.debug(
            "IDW gridding %d points to %d grid nodes "
            "(radius=%.2f, power=%.2f, using KD-tree)...",
            npoints,
            len(xi_flat),
            radius,
            power,
        )

        # For each grid point, use KD-tree to find nearby points
        for i in range(len(xi_flat)):
            idx_nearby = tree.query_ball_point([xi_flat[i], yi_flat[i]], radius)

            if len(idx_nearby) < min_points:
                continue

            # Get nearby points
            x_nearby = xcv[idx_nearby]
            y_nearby = ycv[idx_nearby]
            z_nearby = zcv[idx_nearby]

            # Calculate distances
            dx = x_nearby - xi_flat[i]
            dy = y_nearby - yi_flat[i]
            distances = np.sqrt(dx**2 + dy**2)

            # Handle coincident points
            zero_dist = distances < 1e-10
            if np.any(zero_dist):
                zi_flat[i] = z_nearby[zero_dist][0]
                continue

            # IDW weights
            weights = 1.0 / (distances**power)
            zi_flat[i] = np.sum(weights * z_nearby) / np.sum(weights)
    else:
        # Simple vectorized approach for small datasets without radius
        logger.debug(
            "IDW gridding %d points to %d grid nodes (power=%.2f, using all points)...",
            npoints,
            len(xi_flat),
            power,
        )

        for i in range(len(xi_flat)):
            # Calculate distances from grid point to all data points
            dx = xcv - xi_flat[i]
            dy = ycv - yi_flat[i]
            distances = np.sqrt(dx**2 + dy**2)

            z_nearby = zcv

            # Skip if not enough points
            if len(distances) < min_points:
                continue

            # Handle coincident points (distance = 0)
            zero_dist = distances < 1e-10
            if np.any(zero_dist):
                zi_flat[i] = z_nearby[zero_dist][0]
                continue

            # Calculate weights: w_i = 1 / d_i^power
            weights = 1.0 / (distances**power)

            # Normalize weights and calculate weighted average
            zi_flat[i] = np.sum(weights * z_nearby) / np.sum(weights)

    # Reshape back to grid
    znew = zi_flat.reshape(xiv.shape)

    logger.debug("IDW gridding ... DONE")

    self._ensure_correct_values(znew)


def _range_to_len_scale(
    range_value: float | tuple[float, float],
    variogram_model: str,
    alpha: float | None = None,
) -> float | tuple[float, float]:
    """
    For kriging. Convert RMS/Petrel-style 'range' to GSTools 'len_scale'.

    Args:
        range_value: Range parameter (distance to sill). Can be:
            - float: isotropic range
            - tuple of 2 floats: anisotropic range_value (range_x, range_y)
        variogram_model: Model name
        alpha: Shape parameter for Stable model (1.0-2.0)

    Returns:
        len_scale for GSTools (float or tuple)
    """
    model = variogram_model.lower()

    # Determine conversion factor based on model
    if model == "gaussian":
        factor = np.sqrt(3)
    elif model == "exponential":
        factor = 3.0
    elif model == "spherical":
        factor = 1.0
    elif model == "linear":
        # The 'linear' doesn't have a traditional range/sill; use the range_value as-is
        factor = 1.0
    elif model == "stable":
        if alpha is None:
            logger.warning("Alpha not provided for Stable model, assuming alpha=1.5")
            alpha = 1.5
        # Linear interpolation between Exponential and Gaussian conversions
        factor = 3.0 - (alpha - 1.0) * (3.0 - np.sqrt(3))
    elif model == "matern":
        factor = 3.0  # assuming nu=0.5 (exponential-like)
    else:
        logger.warning(f"Unknown conversion for {variogram_model}, using range_value/2")
        factor = 2.0

    # Apply conversion
    if isinstance(range_value, (tuple, list)):
        # Anisotropic case
        if len(range_value) != 2:
            raise ValueError(
                f"Range tuple must have 2 elements, got {len(range_value)}"
            )
        return tuple(r / factor for r in range_value)

    # Isotropic case
    return range_value / factor


def _create_variogram_model(
    variogram_model: str,
    len_scale: float | tuple[float, float],
    nugget: float,
    variance: float,
    variogram_parameters: dict[str, Any] | None,
    angle: float | None = None,
) -> gs.CovModel:
    """
    Create GSTools covariance model for kriging.

    Args:
        variogram_model: Model name (e.g., 'gaussian', 'exponential')
        len_scale: Correlation length
        nugget: Nugget effect
        variance: Variance/sill
        variogram_parameters: Dict with additional model-specific parameters
        angle: Rotation angle in degrees for anisotropy (GSTools convention).
            In 2D, angle is measured counter-clockwise from East (X-axis).
            Default None means no rotation (0 degrees).

    Returns:
        GSTools covariance model instance
    """

    # Create covariance model
    model_params = {
        "dim": 2,
        "var": variance,
        "len_scale": len_scale,
        "nugget": nugget,
    }

    # Add rotation angle if specified (convert to radians for GSTools)
    # Note: GSTools uses 'angles' parameter name in 2D (even for a single angle)
    if angle is not None:
        model_params["angles"] = np.deg2rad(angle)

    variogram_model = variogram_model.capitalize()  # GStools use capitalized name
    vario_params = variogram_parameters or {}

    # Add model-specific parameters
    if variogram_model == "Matern":
        model_params["nu"] = vario_params.get("nu", 0.5)
    elif variogram_model == "Stable":
        model_params["alpha"] = vario_params.get("alpha", 2.0)

    # Get the model class from gstools
    model_class = getattr(gs, variogram_model, None)
    if model_class is None:
        raise ValueError(
            f"Unknown variogram model: {variogram_model.lower()}. "
            "Valid options: gaussian, exponential, spherical, matern, "
            "stable, linear"
        )

    return model_class(**model_params)


def _calculate_block_division(
    self: RegularSurface, target_cells_per_block: int
) -> tuple[int, int]:
    """
    Calculate optimal block division respecting grid aspect ratio, for kriging.

    Args:
        self: RegularSurface instance
        target_cells_per_block: Target number of cells per block

    Returns:
        tuple: (n_blocks_x, n_blocks_y)
    """
    aspect_ratio = self.ncol / self.nrow
    if aspect_ratio > 1:
        # Wider than tall
        n_blocks_y = max(
            1,
            int(np.sqrt(self.nrow * self.ncol / target_cells_per_block / aspect_ratio)),
        )
        n_blocks_x = max(1, int(n_blocks_y * aspect_ratio))
    else:
        # Taller than wide
        n_blocks_x = max(
            1,
            int(np.sqrt(self.ncol * self.nrow / target_cells_per_block * aspect_ratio)),
        )
        n_blocks_y = max(1, int(n_blocks_x / aspect_ratio))

    # Limit to reasonable values
    n_blocks_x = min(20, max(1, n_blocks_x))
    n_blocks_y = min(20, max(1, n_blocks_y))

    return n_blocks_x, n_blocks_y


def _krige_single_block(
    self: RegularSurface,
    i_block: int,
    j_block: int,
    n_blocks_x: int,
    n_blocks_y: int,
    xiv: FloatArray,
    yiv: FloatArray,
    xcv: FloatArray,
    ycv: FloatArray,
    zcv: FloatArray,
    tree: cKDTree,
    model: gs.CovModel,
    krige_type: Literal["ordinary", "simple"],
    mean: float | None,
    max_points: int,
    margin: float,
) -> tuple[
    FloatArray | None,
    int,
    int,
    int,
    int,
    Literal["success", "empty", "masked", "no_valid", "fallback", "nearest_neighbor"],
]:
    """
    Krige a single block of the grid.

    Args:
        self: RegularSurface instance
        i_block, j_block: Block indices
        n_blocks_x, n_blocks_y: Total number of blocks
        xiv, yiv: Grid coordinate arrays
        xcv, ycv, zcv: Point data coordinates and values
        tree: KD-tree of point data
        model: GSTools covariance model
        krige_type: 'ordinary' or 'simple'
        mean: Mean value for simple kriging
        max_points: Maximum points to use for kriging
        margin: Search margin around block

    Returns:
        tuple: (block_result, i_start, i_end, j_start, j_end, status)
        where status is one of: 'success', 'empty', 'masked', 'no_valid', 'fallback'
    """

    # Calculate block bounds in index space
    # Note: array is shaped (ncol, nrow), so first index is i (columns)
    i_start = i_block * self.ncol // n_blocks_x
    if i_block < n_blocks_x - 1:
        i_end = (i_block + 1) * self.ncol // n_blocks_x
    else:
        i_end = self.ncol
    j_start = j_block * self.nrow // n_blocks_y
    if j_block < n_blocks_y - 1:
        j_end = (j_block + 1) * self.nrow // n_blocks_y
    else:
        j_end = self.nrow

    # Get grid coordinates for this block
    # Array indexing: [i (ncol dimension), j (nrow dimension)]
    xiv_block = xiv[i_start:i_end, j_start:j_end]
    yiv_block = yiv[i_start:i_end, j_start:j_end]

    # Skip truly empty blocks (shouldn't happen with proper slicing)
    if xiv_block.size == 0 or yiv_block.size == 0:
        logger.warning(
            "Block (%d, %d) is empty: size=(%d, %d)",
            i_block,
            j_block,
            xiv_block.size,
            yiv_block.size,
        )
        return None, i_start, i_end, j_start, j_end, "empty"

    # Handle masked arrays - get valid (non-masked) data for block bounds
    # But we still want to krige ALL positions, even masked ones
    if np.ma.is_masked(xiv_block):
        xiv_valid = xiv_block.compressed()
        yiv_valid = yiv_block.compressed()
        if len(xiv_valid) == 0:
            # Block is entirely masked - skip it as it has no coordinates
            return None, i_start, i_end, j_start, j_end, "masked"
        # Use valid data to determine block center for point search
        x_block_min, x_block_max = xiv_valid.min(), xiv_valid.max()
        y_block_min, y_block_max = yiv_valid.min(), yiv_valid.max()

        # For kriging target, use ALL positions (unmasked the data)
        # We want to fill the masked regions with kriged values
        if hasattr(xiv_block, "data"):
            xiv_block_data = xiv_block.data
            yiv_block_data = yiv_block.data
        else:
            xiv_block_data = xiv_block
            yiv_block_data = yiv_block
    else:
        x_block_min, x_block_max = xiv_block.min(), xiv_block.max()
        y_block_min, y_block_max = yiv_block.min(), yiv_block.max()
        xiv_block_data = xiv_block
        yiv_block_data = yiv_block

    # Calculate block center for spatial search
    x_block_center = (x_block_min + x_block_max) / 2
    y_block_center = (y_block_min + y_block_max) / 2
    block_radius = (
        np.sqrt((x_block_max - x_block_min) ** 2 + (y_block_max - y_block_min) ** 2) / 2
        + margin
    )

    # Find points within search radius of block
    idx_near = tree.query_ball_point([x_block_center, y_block_center], block_radius)

    # Flatten block grid coordinates - use unmasked data for kriging
    # We want to krige ALL positions, filling in masked/undefined areas
    x_block_flat = xiv_block_data.ravel()
    y_block_flat = yiv_block_data.ravel()

    # Check for and remove any NaN coordinates (invalid grid positions)
    valid_coords = ~(np.isnan(x_block_flat) | np.isnan(y_block_flat))
    if not valid_coords.any():
        return None, i_start, i_end, j_start, j_end, "no_valid"

    x_krige = x_block_flat[valid_coords]
    y_krige = y_block_flat[valid_coords]

    # Handle case with too few points - use nearest neighbor fallback
    if len(idx_near) < 3:
        _, idx_nearest = tree.query(np.column_stack([x_krige, y_krige]), k=1)
        block_result = np.full(xiv_block.shape, np.nan)
        block_result_flat = block_result.ravel()
        block_result_flat[valid_coords] = zcv[idx_nearest]
        block_result = block_result_flat.reshape(xiv_block.shape)
        return block_result, i_start, i_end, j_start, j_end, "nearest_neighbor"

    # Limit to max_points if we have too many
    if len(idx_near) > max_points:
        # Take closest max_points
        dists = np.sqrt(
            (xcv[idx_near] - x_block_center) ** 2
            + (ycv[idx_near] - y_block_center) ** 2
        )
        closest_idx = np.argsort(dists)[:max_points]
        idx_near = [idx_near[k] for k in closest_idx]

    # Extract local points for this block
    xcv_local = xcv[idx_near]
    ycv_local = ycv[idx_near]
    zcv_local = zcv[idx_near]

    # Perform kriging for this block
    try:
        if krige_type == "ordinary":
            krige = gs.krige.Ordinary(
                model=model,
                cond_pos=[xcv_local, ycv_local],
                cond_val=zcv_local,
            )
        else:
            krige = gs.krige.Simple(
                model=model,
                cond_pos=[xcv_local, ycv_local],
                cond_val=zcv_local,
                mean=mean,
            )

        z_krige_result, _ = krige.unstructured([x_krige, y_krige])

        # Fill results - create array and fill valid coordinate positions
        block_result = np.full(xiv_block.shape, np.nan)
        block_result_flat = block_result.ravel()
        block_result_flat[valid_coords] = z_krige_result
        block_result = block_result_flat.reshape(xiv_block.shape)

        return block_result, i_start, i_end, j_start, j_end, "success"

    except Exception as err:
        logger.warning(
            "Kriging failed for block (%d, %d), using IDW fallback: %s",
            i_block,
            j_block,
            err,
        )
        # Fallback to inverse distance weighting
        z_idw = np.full(len(x_krige), np.nan)
        for idx in range(len(x_krige)):
            dists = np.sqrt(
                (xcv_local - x_krige[idx]) ** 2 + (ycv_local - y_krige[idx]) ** 2
            )
            weights = 1.0 / (dists**2 + 1e-10)
            z_idw[idx] = np.sum(weights * zcv_local) / np.sum(weights)

        # Fill results
        block_result = np.full(xiv_block.shape, np.nan)
        block_result_flat = block_result.ravel()
        block_result_flat[valid_coords] = z_idw
        block_result = block_result_flat.reshape(xiv_block.shape)

        return block_result, i_start, i_end, j_start, j_end, "fallback"


def _block_kriging(
    self: RegularSurface,
    xcv: FloatArray,
    ycv: FloatArray,
    zcv: FloatArray,
    xiv: FloatArray,
    yiv: FloatArray,
    model: gs.CovModel,
    krige_type: Literal["ordinary", "simple"],
    mean: float | None,
    max_points: int,
    len_scale: float | tuple[float, float],
    radius: float | None = None,
) -> FloatArray | MaskedArray:
    """
    Perform block-based kriging for large datasets.

    Args:
        self: RegularSurface instance
        xcv, ycv, zcv: Point data coordinates and values
        xiv, yiv: Grid coordinate arrays
        model: GSTools covariance model
        krige_type: 'ordinary' or 'simple'
        mean: Mean value for simple kriging
        max_points: Maximum points per block
        len_scale: Correlation length for margin calculation
        radius: Search radius override (if None, uses 2 * len_scale)

    Returns:
        numpy array: Kriged surface values
    """
    npoints = len(xcv)

    logger.debug(
        "Using block-based kriging with max %d points per block (total: %d points)...",
        max_points,
        npoints,
    )

    # Build KD-tree for fast spatial searches
    tree = cKDTree(np.column_stack([xcv, ycv]))

    # Determine block size based on grid dimensions
    total_cells = self.ncol * self.nrow
    target_cells_per_block = max(100, total_cells // 100)

    n_blocks_x, n_blocks_y = _calculate_block_division(self, target_cells_per_block)
    logger.debug("Dividing grid into %d x %d blocks...", n_blocks_x, n_blocks_y)

    # Initialize result array - preserve masking if input has it
    if np.ma.is_masked(xiv):
        znew = np.ma.array(
            np.full(xiv.shape, np.nan),
            mask=xiv.mask.copy() if hasattr(xiv, "mask") else False,
        )
    else:
        znew = np.full(xiv.shape, np.nan)

    # Calculate search margin for finding points around each block
    # Use provided radius or auto-calculate from correlation length
    if radius is not None:
        margin = radius
        logger.debug("Using provided search radius: %.1f", margin)
    else:
        if isinstance(len_scale, float):
            margin = 2.0 * len_scale
        else:
            margin = 2.0 * np.max(len_scale)
        logger.debug("Auto-calculated search radius: %.1f (2 * len_scale)", margin)

    # Process each block
    block_count = 0
    skipped_empty = 0
    skipped_masked = 0
    skipped_no_valid = 0

    for i_block in range(n_blocks_x):
        for j_block in range(n_blocks_y):
            result, i_start, i_end, j_start, j_end, status = _krige_single_block(
                self,
                i_block,
                j_block,
                n_blocks_x,
                n_blocks_y,
                xiv,
                yiv,
                xcv,
                ycv,
                zcv,
                tree,
                model,
                krige_type,
                mean,
                max_points,
                margin,
            )

            if status == "empty":
                skipped_empty += 1
            elif status == "masked":
                skipped_masked += 1
            elif status == "no_valid":
                skipped_no_valid += 1
            elif result is not None:
                znew[i_start:i_end, j_start:j_end] = result
                if status in ("success", "fallback"):
                    block_count += 1

    total_blocks = n_blocks_x * n_blocks_y
    logger.debug(
        "Block kriging completed: %d/%d blocks processed "
        "(skipped: %d empty, %d masked, %d no valid cells)",
        block_count,
        total_blocks,
        skipped_empty,
        skipped_masked,
        skipped_no_valid,
    )

    # Check how much of the surface was filled
    if np.ma.is_masked(znew):
        n_valid = (~znew.mask).sum()
        n_nan = np.isnan(znew.data[~znew.mask]).sum() if n_valid > 0 else 0
    else:
        n_valid = znew.size
        n_nan = np.isnan(znew).sum()

    logger.debug(
        "Surface coverage: %d valid cells, %d NaN cells (%.1f%% coverage)",
        n_valid - n_nan,
        n_nan,
        100.0 * (n_valid - n_nan) / n_valid if n_valid > 0 else 0,
    )

    return znew


def _global_kriging(
    self: RegularSurface,
    xcv: FloatArray,
    ycv: FloatArray,
    zcv: FloatArray,
    xiv: FloatArray,
    yiv: FloatArray,
    model: gs.CovModel,
    krige_type: Literal["ordinary", "simple"],
    mean: float | None,
) -> FloatArray:
    """
    Perform global kriging using all points.

    Args:
        self: RegularSurface instance
        xcv, ycv, zcv: Point data coordinates and values
        xiv, yiv: Grid coordinate arrays
        model: GSTools covariance model
        krige_type: 'ordinary' or 'simple'
        mean: Mean value for simple kriging

    Returns:
        numpy array: Kriged surface values
    """

    x_coords = xiv.ravel()
    y_coords = yiv.ravel()

    if krige_type == "ordinary":
        krige = gs.krige.Ordinary(
            model=model,
            cond_pos=[xcv, ycv],
            cond_val=zcv,
        )
    else:
        krige = gs.krige.Simple(
            model=model,
            cond_pos=[xcv, ycv],
            cond_val=zcv,
            mean=mean,
        )

    znew_flat, sigma2_flat = krige.unstructured([x_coords, y_coords])
    return znew_flat.reshape(xiv.shape)


def _enforce_exact_bilinear(
    self: RegularSurface,
    xcv: FloatArray,
    ycv: FloatArray,
    zcv: FloatArray,
    znew: FloatArray,
) -> int:
    """
    Enforce exact values using bilinear interpolation adjustment.

    For each data point:
    1. Find the 4 surrounding grid nodes
    2. Calculate bilinear weights at the point location
    3. Adjust the 4 node values to ensure bilinear surface = data value

    This is more accurate than "nearest" because:
    - Points are honored at their exact (x,y) location, not snapped to nodes
    - Works for points anywhere in grid cell (not just at nodes)
    - Smoother transitions when multiple nearby points

    Args:
        self: RegularSurface instance
        xcv, ycv, zcv: Point data coordinates and values
        znew: Kriged surface values (modified in-place)

    Returns:
        int: Number of data points enforced
    """
    # Get grid coordinate arrays - needed to find cells for rotated surfaces
    xiv, yiv = self.get_xy_values()

    # Build KD-tree of all grid nodes for fast nearest-neighbor search
    # Flatten arrays to 1D: array is (ncol, nrow), so ravel gives ncol*nrow points
    x_nodes = xiv.ravel()
    y_nodes = yiv.ravel()
    tree = cKDTree(np.column_stack([x_nodes, y_nodes]))

    n_enforced = 0
    n_outside = 0
    n_tested = 0

    for k in range(len(xcv)):
        x, y, z = xcv[k], ycv[k], zcv[k]

        # Find nearest grid node as starting point
        _, nearest_idx = tree.query([x, y], k=1)

        # Convert flat index back to (i, j) grid indices
        # Array is shaped (ncol, nrow), stored in C-order
        i_nearest = nearest_idx // self.nrow
        j_nearest = nearest_idx % self.nrow

        # Search in a small neighborhood to find the cell containing the point
        # Check cells around the nearest node - expand search if needed
        found_cell = False
        n_tested += 1

        # Try expanding search radius if point not found
        for search_radius in [0, 1, 2]:
            if found_cell:
                break

            for di in range(-search_radius, search_radius + 1):
                for dj in range(-search_radius, search_radius + 1):
                    i0 = i_nearest + di
                    j0 = j_nearest + dj

                    # Check if this cell is valid
                    if i0 < 0 or i0 >= self.ncol - 1 or j0 < 0 or j0 >= self.nrow - 1:
                        continue

                    i1 = i0 + 1
                    j1 = j0 + 1

                    # Get the 4 corner coordinates
                    # Grid cells: (i0,j0) is one corner, (i1,j1) is opposite
                    # Order counter-clockwise: bottom-left, bottom-right,
                    # top-right, top-left
                    x00, y00 = xiv[i0, j0], yiv[i0, j0]  # bottom-left
                    x10, y10 = xiv[i1, j0], yiv[i1, j0]  # bottom-right (i+)
                    x11, y11 = xiv[i1, j1], yiv[i1, j1]  # top-right
                    x01, y01 = xiv[i0, j1], yiv[i0, j1]  # top-left (j+)

                    # Check if point is inside this quadrilateral
                    def point_in_quad(px, py, x0, y0, x1, y1, x2, y2, x3, y3):
                        """
                        Check if point (px, py) is inside quad with corners
                        ordered as: (x0,y0) -> (x1,y1) -> (x2,y2) -> (x3,y3)
                        """

                        # Check if point is on same side of each edge
                        def cross_sign(ax, ay, bx, by, px, py):
                            return (bx - ax) * (py - ay) - (by - ay) * (px - ax)

                        # Check edges: 0->1, 1->2, 2->3, 3->0
                        s0 = cross_sign(x0, y0, x1, y1, px, py)
                        s1 = cross_sign(x1, y1, x2, y2, px, py)
                        s2 = cross_sign(x2, y2, x3, y3, px, py)
                        s3 = cross_sign(x3, y3, x0, y0, px, py)

                        # All should have same sign (or be zero)
                        return (s0 >= 0 and s1 >= 0 and s2 >= 0 and s3 >= 0) or (
                            s0 <= 0 and s1 <= 0 and s2 <= 0 and s3 <= 0
                        )

                    # Pass corners in order: BL, BR, TR, TL
                    if point_in_quad(x, y, x00, y00, x10, y10, x11, y11, x01, y01):
                        found_cell = True

                        # Calculate bilinear interpolation weights using
                        # inverse transformation. For a bilinear surface:
                        # f(s,t) = (1-s)(1-t)f00 + s(1-t)f10 + (1-s)t*f01
                        #          + st*f11
                        # where s, t are parametric coordinates in [0,1]
                        # We need to find s, t such that coords match

                        # Use iterative Newton-Raphson to find (s, t)
                        # from (x, y). Needed for general quads (rotated
                        # grids)
                        s, t = 0.5, 0.5  # Initial guess at cell center

                        for _ in range(10):  # Newton iterations
                            # Current position
                            x_curr = (
                                (1 - s) * (1 - t) * x00
                                + s * (1 - t) * x10
                                + (1 - s) * t * x01
                                + s * t * x11
                            )
                            y_curr = (
                                (1 - s) * (1 - t) * y00
                                + s * (1 - t) * y10
                                + (1 - s) * t * y01
                                + s * t * y11
                            )

                            # Residual
                            rx = x - x_curr
                            ry = y - y_curr

                            if abs(rx) < 1e-6 and abs(ry) < 1e-6:
                                break

                            # Jacobian matrix
                            dx_ds = (1 - t) * (x10 - x00) + t * (x11 - x01)
                            dx_dt = (1 - s) * (x01 - x00) + s * (x11 - x10)
                            dy_ds = (1 - t) * (y10 - y00) + t * (y11 - y01)
                            dy_dt = (1 - s) * (y01 - y00) + s * (y11 - y10)

                            det = dx_ds * dy_dt - dx_dt * dy_ds
                            if abs(det) < 1e-10:
                                break

                            # Newton step
                            ds = (dy_dt * rx - dx_dt * ry) / det
                            dt = (-dy_ds * rx + dx_ds * ry) / det

                            s += ds
                            t += dt

                            # Clamp to valid range
                            s = np.clip(s, 0, 1)
                            t = np.clip(t, 0, 1)

                        # Calculate bilinear weights
                        w00 = (1 - s) * (1 - t)
                        w10 = s * (1 - t)
                        w01 = (1 - s) * t
                        w11 = s * t

                        # Current interpolated value
                        z_current = (
                            w00 * znew[i0, j0]
                            + w10 * znew[i1, j0]
                            + w01 * znew[i0, j1]
                            + w11 * znew[i1, j1]
                        )

                        # Calculate and distribute adjustment
                        z_adjust = z - z_current
                        znew[i0, j0] += w00 * z_adjust
                        znew[i1, j0] += w10 * z_adjust
                        znew[i0, j1] += w01 * z_adjust
                        znew[i1, j1] += w11 * z_adjust

                        n_enforced += 1
                        break

                if found_cell:
                    break

        if not found_cell:
            n_outside += 1

    logger.debug(
        "Enforced exact values for %d/%d points using bilinear adjustment "
        "(%d points outside grid)",
        n_enforced,
        n_tested,
        n_outside,
    )
    return n_enforced


def points_kriging_gridding(
    self: RegularSurface,
    points: Points,
    variogram_model: str = "gaussian",
    variogram_parameters: dict[str, Any] | None = None,
    krige_type: Literal["ordinary", "simple"] = "ordinary",
    mean: float | None = None,
    coarsen: int = 1,
    max_points: int = 500,
    radius: float | None = None,
    exact: bool = False,
) -> None:
    """
    Do kriging gridding from a points data set using GSTools.

    Kriging provides optimal spatial interpolation based on spatial correlation
    structure (variogram). Good for sparse, spatially correlated data.

    Args:
        points: Points object containing x, y, z coordinates
        variogram_model: Variogram/Covariance model. Options:
            - 'gaussian': Gaussian model (smooth, good for continuous phenomena)
            - 'exponential': Exponential model (good for geology)
            - 'spherical': Spherical model (classic geostatistics)
            - 'matern': Matérn model (flexible smoothness)
            - 'stable': Stable model (generalization of Gaussian, also called
              'general exponential')
            - 'linear': Linear model
        variogram_parameters: Dict with variogram model parameters:
            - 'len_scale' (float or tuple): Correlation length (range). If None,
              auto-estimated. Can be single value or tuple (len_scale_x, len_scale_y)
              for anisotropy. First value is major axis (X or Easting), second is
              minor axis (Y or Northing).
            - 'range_value' (float or tuple): Alternative to len_scale (RMS/Petrel
              style). If both provided, len_scale takes precedence. Can be single
              value or tuple for anisotropy.
            - 'angle' (float): Rotation angle in degrees for anisotropic variograms
              (counter-clockwise from East/X-axis). Only used when len_scale or
              range_value is a tuple. Default None (0°).
              Examples: 0°=East, 90°=North, 45°=NE, -45°=SE.
            - 'nugget' (float): Nugget effect (measurement error/micro-scale
              variation), default 0.0
            - 'variance' (float): Variance/sill of the model. If None, auto-estimated
              from data variance.
            - 'nu' (float): Smoothness parameter for Matern model, default 0.5
            - 'alpha' (float): Shape parameter for Stable model, default 2.0
        krige_type: Type of kriging, 'ordinary' (default) or 'simple'
        mean: Mean value for simple kriging. If None and krige_type='simple',
            auto-estimated from data. Not used for ordinary kriging.
        coarsen: Coarsening factor for input points
        max_points: Maximum points per block for kriging. When dataset has more
            points, uses hybrid block-based kriging: divides grid into blocks and
            uses nearby points for each block. This balances speed and accuracy
            better than simple subsampling. Recommended: 500-2000.
            If None, defaults to 500. Use a very large value to force global
            kriging with all points.
        radius: Search radius (in map units) for finding points around each block.
            If None, automatically calculated as 2 * len_scale to ensure overlap
            between blocks. Increase if blocks have too few points, decrease for
            faster execution with dense data.
        exact: If True, enforces exact data values at point locations using bilinear
            interpolation adjustment. Default is False.
            Note: Kriging is theoretically exact when nugget=0, but grid nodes
            rarely coincide with data points, so enforcement is recommended.

    Example::
        # Simple isotropic kriging (auto-estimated parameters)
        surf.gridding(points, method='kriging')

        # Isotropic with explicit range
        surf.gridding(points, method='kriging',
                     method_options={
                         'variogram_model': 'exponential',
                         'variogram_parameters': {'range_value': 1500.0}
                     })

        # Anisotropic kriging with rotation
        surf.gridding(points, method='kriging',
                     method_options={
                         'variogram_model': 'gaussian',
                         'variogram_parameters': {
                             'len_scale': (2000, 500),  # major, minor
                             'angle': 45,  # NE direction
                             'nugget': 0.1
                         }
                     })

        # Full control with all parameters
        surf.gridding(points, method='kriging',
                     method_options={
                         'variogram_model': 'matern',
                         'variogram_parameters': {
                             'range_value': (3000, 1200),
                             'angle': 30,
                             'nugget': 0.05,
                             'variance': 100.0,
                             'nu': 1.5  # Matern-specific
                         },
                         'krige_type': 'simple',
                         'mean': 1700.0,
                         'exact': True
                     })

        # Block-based kriging for large datasets
        surf.gridding(points, method='kriging',
                     method_options={
                         'max_points': 500,
                         'radius': 5000.0
                     })
    """

    logger.debug("Do kriging...")
    dfra, xcv, ycv, zcv, xiv, yiv = _points_gridding_common(
        self, points, coarsen=coarsen
    )

    # Check for close points that may cause numerical issues
    _check_close_points_warning(self, points, threshold_factor=0.1)

    npoints = len(xcv)

    # Extract variogram parameters from dict
    vario_params = variogram_parameters or {}

    len_scale = vario_params.get("len_scale", None)
    range_value = vario_params.get("range_value", None)
    nugget = vario_params.get("nugget", 0.0)
    angle = vario_params.get("angle", None)

    # Validate kriging type
    if krige_type not in ("ordinary", "simple"):
        raise ValueError(
            f"krige_type must be 'ordinary' or 'simple', got '{krige_type}'"
        )

    # For simple kriging, estimate or use provided mean
    if krige_type == "simple":
        if mean is None:
            mean = np.mean(zcv)
            logger.debug("Auto-estimated mean for simple kriging: %.2f", mean)
        else:
            logger.debug("Using provided mean for simple kriging: %.2f", mean)

    # Handle range_value vs len_scale
    if range_value is not None and len_scale is None:
        # Extract alpha for Stable model if needed
        alpha = None
        if variogram_model.lower() == "stable" and variogram_parameters:
            alpha = variogram_parameters.get("alpha", None)

        len_scale = _range_to_len_scale(range_value, variogram_model, alpha=alpha)

        if isinstance(len_scale, tuple):
            logger.debug(
                "Converted RMS range=(%s, %s) to GSTools len_scale=(%s, %s) "
                "for %s model%s",
                range_value[0],
                range_value[1],
                len_scale[0],
                len_scale[1],
                variogram_model,
                f" (alpha={alpha})" if alpha is not None else "",
            )
        else:
            logger.debug(
                "Converted RMS range=%.1f to GSTools len_scale=%.1f for %s model%s",
                range_value,
                len_scale,
                variogram_model,
                f" (alpha={alpha})" if alpha is not None else "",
            )

    # Auto-estimate len_scale if not provided
    if len_scale is None:
        len_scale = _estimate_correlation_length(xcv, ycv, zcv)
        logger.debug("Auto-estimated correlation length: %.2f", len_scale)

    # Extract variance if provided
    variance = vario_params.get("variance", None)

    # Auto-estimate variance if not provided
    if variance is None:
        variance = np.var(zcv)
        logger.debug("Auto-estimated variance: %.2f", variance)

    # Create covariance model
    model = _create_variogram_model(
        variogram_model, len_scale, nugget, variance, variogram_parameters, angle
    )

    # Determine kriging strategy based on max_points and dataset size
    if max_points is None:
        max_points = 500

    use_block_kriging = npoints > max_points

    if use_block_kriging:
        # Hybrid block-based approach
        znew = _block_kriging(
            self,
            xcv,
            ycv,
            zcv,
            xiv,
            yiv,
            model,
            krige_type,
            mean,
            max_points,
            len_scale,
            radius,
        )
    else:
        # Global kriging - use all points (fast for small datasets)
        logger.debug(
            "Global kriging from %d points (type=%s, model=%s, len_scale=%s, "
            "nugget=%.2f, var=%.2f)...",
            npoints,
            krige_type,
            variogram_model,
            len_scale,
            nugget,
            variance,
        )
        znew = _global_kriging(self, xcv, ycv, zcv, xiv, yiv, model, krige_type, mean)

    # Enforce exact values at data points
    if exact:
        logger.debug("Enforcing exact data values...")
        n_enforced = _enforce_exact_bilinear(self, xcv, ycv, zcv, znew)
        if n_enforced > 0:
            logger.debug(
                "Enforced exact values at %d locations using '%s' method",
                n_enforced,
                exact,
            )

    logger.debug("Kriging gridding ... DONE")
    self._ensure_correct_values(znew)


# ======================================================================================
# FUNCTIONS doing gridding from 3D properties
# ======================================================================================


def avgsum_from_3dprops_gridding(
    self: RegularSurface,
    summing: bool = False,
    xprop: Any = None,
    yprop: Any = None,
    mprop: Any = None,
    dzprop: Any = None,
    truncate_le: Any = None,
    zoneprop: Any = None,
    zone_minmax: tuple[int, int] | None = None,
    coarsen: int = 1,
    zone_avg: bool = False,
    mask_outside: bool = False,
) -> None:
    """Get surface average from a 3D grid prop."""
    # NOTE:
    # This do _either_ averaging _or_ sum gridding (if summing is True)
    # - Inputs shall be pure 3D numpies, not masked!
    # - Xprop and yprop must be made for all cells
    # - Also dzprop for all cells, and dzprop = 0 for inactive cells!

    logger.debug("Avgsum calculation %s", __name__)

    if zone_minmax is None:
        raise ValueError("zone_minmax is required")

    if dzprop is None:
        raise ValueError("DZ property is required")

    xprop, yprop, zoneprop, mprop, dzprop = _zone_averaging(
        xprop,
        yprop,
        zoneprop,
        zone_minmax,
        coarsen,
        zone_avg,
        dzprop,
        mprop,
        summing=summing,
    )

    gnlay = xprop.shape[2]

    # avoid artifacts from inactive cells that slips through somehow...(?)
    if dzprop.max() > UNDEF_LIMIT:
        raise RuntimeError("Bug: DZ with unphysical values present")

    trimbydz = False
    if not summing:
        trimbydz = True

    if summing and mask_outside:
        trimbydz = True

    xiv, yiv = self.get_xy_values()

    # weight are needed if zoneprop is not follow layers, but rather regions
    weights = dzprop.copy() * 0.0 + 1.0
    weights[zoneprop < zone_minmax[0]] = 0.0
    weights[zoneprop > zone_minmax[1]] = 0.0

    # this operation is needed if zoneprop is aka a region ("irregular zone")
    zoneprop = ma.masked_less(zoneprop, zone_minmax[0])
    zoneprop = ma.masked_greater(zoneprop, zone_minmax[1])

    for klay0 in range(gnlay):
        k1lay = klay0 + 1

        if k1lay == 1:
            msum = np.zeros((self.ncol, self.nrow), order="C")
            dzsum = np.zeros((self.ncol, self.nrow), order="C")

        numz = zoneprop[::, ::, klay0].mean()
        if isinstance(numz, float):
            numz = int(round(zoneprop[::, ::, klay0].mean()))
            if numz < zone_minmax[0] or numz > zone_minmax[1]:
                continue
        else:
            continue

        qmcompute = True
        if summing:
            propsum = mprop[:, :, klay0].sum()
            if abs(propsum) < 1e-12:
                logger.debug("Too little HC, skip layer K = %s", k1lay)
                qmcompute = False
            else:
                logger.debug("Z property sum is %s", propsum)

        logger.debug("Mapping for layer or zone %s ....", k1lay)

        xcv = xprop[::, ::, klay0].ravel(order="C")
        ycv = yprop[::, ::, klay0].ravel(order="C")
        mvv = mprop[::, ::, klay0].ravel(order="C")
        dzv = dzprop[::, ::, klay0].ravel(order="C")
        wei = weights[::, ::, klay0].ravel(order="C")

        # this is done to avoid problems if undef values still remains
        # in the coordinates (assume Y undef where X undef):
        xcc = xcv.copy()
        xcv = xcv[xcc < 1e20]
        ycv = ycv[xcc < 1e20]
        mvv = mvv[xcc < 1e20]
        dzv = dzv[xcc < 1e20]
        wei = wei[xcc < 1e20]

        mvdz = mvv * wei if summing else mvv * dzv * wei

        if qmcompute:
            try:
                mvdzi = scipy.interpolate.griddata(
                    (xcv, ycv), mvdz, (xiv, yiv), method="linear", fill_value=0.0
                )
            except ValueError:
                warnings.warn("Some problems in gridding ... will contue", UserWarning)
                continue

            msum = msum + mvdzi

        if trimbydz:
            try:
                dzi = scipy.interpolate.griddata(
                    (xcv, ycv), dzv, (xiv, yiv), method="linear", fill_value=0.0
                )
            except ValueError:
                continue

            dzsum = dzsum + dzi

    if not summing:
        dzsum[dzsum == 0.0] = 1e-20
        vvz = msum / dzsum
        vvz = ma.masked_invalid(vvz)
    else:
        vvz = msum

    vvz = (
        ma.masked_where(dzsum < 1.1e-20, vvz) if trimbydz else ma.array(vvz)
    )  # so the result becomes a ma array

    if truncate_le:
        vvz = ma.masked_less(vvz, truncate_le)

    self.values = vvz
    logger.debug("Avgsum calculation done! %s", __name__)

    return True


def _zone_averaging(
    xprop: Any,
    yprop: Any,
    zoneprop: Any,
    zone_minmax: tuple[int, int] | None,
    coarsen: int,
    zone_avg: bool,
    dzprop: Any,
    mprop: Any,
    summing: bool = False,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    # General preprocessing, and...
    # Change the 3D numpy array so they get layers by
    # averaging across zones. This may speed up a lot,
    # but will reduce the resolution.
    # The x y coordinates shall be averaged (ideally
    # with thickness weighting...) while e.g. hcpfzprop
    # must be summed.
    # Somewhat different processing whether this is a hc thickness
    # or an average.

    xpr = xprop
    ypr = yprop
    zpr = zoneprop
    dpr = dzprop

    mpr = mprop

    if coarsen > 1:
        xpr = xprop[::coarsen, ::coarsen, ::].copy(order="C")
        ypr = yprop[::coarsen, ::coarsen, ::].copy(order="C")
        zpr = zoneprop[::coarsen, ::coarsen, ::].copy(order="C")
        dpr = dzprop[::coarsen, ::coarsen, ::].copy(order="C")
        mpr = mprop[::coarsen, ::coarsen, ::].copy(order="C")
        zpr.astype(np.int32)

    if zone_avg:
        zmin = int(zone_minmax[0])
        zmax = int(zone_minmax[1])
        if zpr.min() > zmin:
            zmin = zpr.min()
        if zpr.max() < zmax:
            zmax = zpr.max()

        newx = []
        newy = []
        newz = []
        newm = []
        newd = []

        for izv in range(zmin, zmax + 1):
            logger.debug("Averaging for zone %s ...", izv)
            xpr2 = ma.masked_where(zpr != izv, xpr)
            ypr2 = ma.masked_where(zpr != izv, ypr)
            zpr2 = ma.masked_where(zpr != izv, zpr)
            dpr2 = ma.masked_where(zpr != izv, dpr)
            mpr2 = ma.masked_where(zpr != izv, mpr)

            # get the thickness and normalize along axis 2 (vertical)
            # to get normalized thickness weights
            lay_sums = dpr2.sum(axis=2)
            normed_dz = dpr2 / lay_sums[:, :, np.newaxis]

            # assume that coordinates have equal weights within a zone
            xpr2 = ma.average(xpr2, axis=2)
            ypr2 = ma.average(ypr2, axis=2)
            zpr2 = ma.average(zpr2, axis=2)  # avg zone

            dpr2 = ma.sum(dpr2, axis=2)

            if summing:
                mpr2 = ma.sum(mpr2, axis=2)
            else:
                mpr2 = ma.average(mpr2, weights=normed_dz, axis=2)  # avg zone

            newx.append(xpr2)
            newy.append(ypr2)
            newz.append(zpr2)
            newd.append(dpr2)
            newm.append(mpr2)

        xpr = ma.dstack(newx)
        ypr = ma.dstack(newy)
        zpr = ma.dstack(newz)
        dpr = ma.dstack(newd)
        mpr = ma.dstack(newm)
        zpr.astype(np.int32)

    xpr = ma.filled(xpr, fill_value=UNDEF)
    ypr = ma.filled(ypr, fill_value=UNDEF)
    zpr = ma.filled(zpr, fill_value=0)
    dpr = ma.filled(dpr, fill_value=0.0)

    mpr = ma.filled(mpr, fill_value=0.0)

    return xpr, ypr, zpr, mpr, dpr


# ======================================================================================
# FUNCTIONS doing post-processing on surfaces
# ======================================================================================


def surf_fill(
    self: RegularSurface, fill_value: float | None = None, method: str = "nearest"
) -> None:
    """Replace the value of invalid 'data' cells (indicated by 'invalid')
    by the value of the nearest valid data cell, a constant, or smooth extrapolation.

    This is a quite fast method to fill undefined areas of the map.
    The surface values are updated 'in-place'

    Args:
        fill_value: If numeric, fill with this constant value
        method: Extrapolation method if fill_value is None:
            - 'nearest': Use nearest valid cell (fast, blocky)
            - 'linear': Linear interpolation/extrapolation (smooth)
            - 'cubic': Cubic interpolation/extrapolation (very smooth)
            - 'radial_basis': RBF with thin plate spline (smooth, best quality)

    .. versionadded:: 2.1
    .. versionchanged:: 2.6 Added fill_value
    """
    logger.debug(
        "Do fill with method '%s'...", method if fill_value is None else "constant"
    )

    if fill_value is not None:
        if np.isscalar(fill_value) and not isinstance(fill_value, str):
            self.values = ma.filled(self.values, fill_value=float(fill_value))
        else:
            raise ValueError("Keyword fill_value must be int or float")
    else:
        valid_mask = ~ma.getmaskarray(self.values)

        if method == "nearest":
            # Original fast method
            invalid = ~valid_mask
            ind = scipy.ndimage.distance_transform_edt(
                invalid, return_distances=False, return_indices=True
            )
            self._values = self._values[tuple(ind)]

        elif method in ("linear", "cubic", "radial_basis"):
            # Get valid data points
            xiv, yiv = self.get_xy_values()

            xcv = xiv[valid_mask]
            ycv = yiv[valid_mask]
            zcv = self.values[valid_mask]

            # Interpolate/extrapolate to all grid points
            if method in ("linear", "cubic"):
                znew = scipy.interpolate.griddata(
                    (xcv, ycv),
                    zcv,
                    (xiv, yiv),
                    method=method,
                    fill_value=np.nan,  # Will still have NaN at far extrapolation
                )

                # For remaining NaN, fall back to nearest
                remaining_nan = np.isnan(znew)
                if np.any(remaining_nan):
                    znear = scipy.interpolate.griddata(
                        (xcv, ycv), zcv, (xiv, yiv), method="nearest"
                    )
                    znew[remaining_nan] = znear[remaining_nan]

            elif method == "radial_basis":
                # RBF can be slow - optimize for large datasets
                max_points = 5000

                if len(xcv) > max_points:
                    logger.debug(
                        "Sampling %d points from %d for RBF fill", max_points, len(xcv)
                    )
                    step = max(1, len(xcv) // max_points)
                    xcv = xcv[::step][:max_points]
                    ycv = ycv[::step][:max_points]
                    zcv = zcv[::step][:max_points]

                rbf = scipy.interpolate.RBFInterpolator(
                    np.column_stack([xcv, ycv]),
                    zcv,
                    kernel="thin_plate_spline",
                    smoothing=0.0,
                    degree=1,
                )
                znew = rbf(np.column_stack([xiv.ravel(), yiv.ravel()])).reshape(
                    xiv.shape
                )
            self._values = znew

        else:
            raise ValueError(
                f"Invalid method '{method}'. "
                "Valid options: 'nearest', 'linear', 'cubic', 'radial_basis'"
            )

    logger.debug("Do fill... DONE")


def _smooth(
    self: RegularSurface,
    window_function: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    iterations: int = 1,
) -> None:
    """
    Smooth a RegularSurface using a window function.

    Original mask (undefined values) is stored before applying
    smoothing on a filled array. Subsequently the original mask
    is used to restore the undefined values in the output.
    """

    mask = ma.getmaskarray(self.values)

    self.fill()

    smoothed_values = self.values
    for _ in range(iterations):
        smoothed_values = window_function(smoothed_values, mode="nearest")

    self.values = ma.array(smoothed_values, mask=mask)
