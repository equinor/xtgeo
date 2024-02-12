"""Various operations dedicated to Polygons class.

First order functions here are:

* boundary_from_points (input to class method)
* simplify_polygons (instance method)


Functions starting with '_' are local helper functions
"""

from __future__ import annotations

from math import ceil

import numpy as np
import pandas as pd
import shapely
import shapely.geometry as sg
from scipy.spatial import Delaunay, cKDTree
from shapely.ops import polygonize

import xtgeo
from xtgeo.common.log import null_logger

logger = null_logger(__name__)
MINIMUM_NUMBER_POINTS = 4


class BoundaryError(ValueError):
    """Get error on not able to create a boundary."""


def boundary_from_points(points, alpha_factor=1.0, alpha=None, concave=False):
    """From a Point instance, make boundary polygons (generic)."""
    if not isinstance(points, xtgeo.Points):
        raise ValueError("The input points is not an instance of xtgeo.Points")

    if points.nrow < MINIMUM_NUMBER_POINTS:
        logger.info("No. points is %s, need >= %s", points.nrow, MINIMUM_NUMBER_POINTS)
        raise ValueError(
            "Too few points to derive a boundary, present is "
            f"{points.nrow}, need at least {MINIMUM_NUMBER_POINTS}"
        )

    if alpha_factor <= 0.0:
        raise ValueError("The alpha_factor value must be greater than 0.0")

    if alpha is not None and alpha <= 0.0:
        raise ValueError("The alpha value must be greater than 0.0")

    if concave:
        alpha_factor = 0
        alpha = 999  # dummy

    usepoints = points.copy()  # make a copy since points may be filtered

    xvec = usepoints.get_dataframe(copy=False)[usepoints.xname].values
    yvec = usepoints.get_dataframe(copy=False)[usepoints.yname].values
    zvec = usepoints.get_dataframe(copy=False)[usepoints.zname].values

    if alpha is None:
        # use scipy to detect average distance; 30 points should be sufficient
        xyv = np.column_stack((xvec, yvec))
        use_npoints = 30 if xvec.size >= 30 else xvec.size
        kdtree = cKDTree(xyv)
        dist, _ = kdtree.query(xyv, k=use_npoints)
        auto_alpha = np.mean(dist)
        logger.info("Proposed auto alpha is %s", auto_alpha)

        # now try this auto_alpha, and iterate to a gradually larger alpha proposal
        # if still no success in creating a boundary
        try:
            return_values = _create_boundary_polygon(
                xvec, yvec, zvec, auto_alpha * alpha_factor
            )
        except BoundaryError as berr:
            # iterate and propose a new alpha value while raising an exception
            propose_alpha = _propose_new_alpha(
                xvec, yvec, zvec, alpha_factor, auto_alpha
            )
            propose_alpha_factor = alpha_factor * propose_alpha / auto_alpha
            msg = (
                "Your alpha_factor or alpha is too low. Set the alpha_factor to "
                f"approx. {propose_alpha_factor:.3f}\n"
            )
            raise BoundaryError(msg) from berr
    else:
        return_values = _create_boundary_polygon(xvec, yvec, zvec, alpha * alpha_factor)

    # return the class parameters to populate the Polygons instance
    if return_values is None:
        raise BoundaryError("Cannot create a a boundary, too few points?")

    return return_values


def _propose_new_alpha(xvec, yvec, zvec, alpha_factor, alpha):
    """The current combination of alpha_factor and alpha did not get produce a result.

    Do an iteration here and propose a new alpha_value, given that the user insists
    on keeping the current alpha_factor.
    """
    trials = 0
    max_trials = 1000
    multiplier = 1.1

    proposed_alpha = alpha * multiplier
    while trials < max_trials:
        trials += 1
        logger.info("Proposed alpha is %s in trial %s", proposed_alpha, trials)
        try:
            _create_boundary_polygon(xvec, yvec, zvec, proposed_alpha * alpha_factor)
            print("Trial number:", trials)
            return ceil(proposed_alpha)
        except BoundaryError:
            proposed_alpha *= multiplier

    raise RuntimeError(f"Not able to estimate an alpha in {max_trials} iterations!")


def _create_boundary_polygon(
    xvec: np.ndarray,
    yvec: np.ndarray,
    zvec: np.ndarray,
    alpha: float,
) -> list[list[float | int]] | None:
    xy = np.column_stack((xvec, yvec))
    coords = np.column_stack((xy, zvec))

    # return None if too few points
    if len(xy) < MINIMUM_NUMBER_POINTS:
        logger.info("Too few points to derive boundary, need at least 4")
        return None

    edges = _alpha_shape(xy, alpha=alpha)
    if not edges:
        raise BoundaryError(
            "Your alpha or alpha_factor value is too low, try increasing it!"
        )

    pol_coords = _get_polygon_coords_from_edges_shapely(coords, edges)
    if not pol_coords:
        raise BoundaryError("Could not create a valid Polygons instance for boundary.")

    # sort polygons by their size to get the largest polygons first
    pol_coords = sorted(pol_coords, key=len, reverse=True)

    data = []
    for polid, polcoord in enumerate(pol_coords):
        for coord in polcoord:
            data.append(list(coord) + [polid])

    return data


def _alpha_shape(points: np.ndarray, alpha: float) -> set[tuple[int, int]]:
    """Compute the alpha shape (concave hull) of a set of points.

    Args:
        points: np.array of shape (n,2) points.
        alpha: alpha value.
    Returns:
        Set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
            the indices in the points array.
    """

    def add_edge(edges: set[tuple[int, int]], icv: int, jcv: int) -> None:
        """Add an edge between the i-th and j-th points, if not in the list already."""
        if (icv, jcv) in edges or (jcv, icv) in edges:
            # if both neighboring triangles are in shape, it is not a boundary edge
            edges.remove((jcv, icv))
            return
        edges.add((icv, jcv))

    tri = Delaunay(points)

    edges = set()

    # Loop over triangles: ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.simplices:
        pa, pb, pc = points[ia], points[ib], points[ic]

        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-\
        # for-radius-of-circumcircle
        avv = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        bvv = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        cvv = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)

        svv = (avv + bvv + cvv) / 2.0
        partial = svv * (svv - avv) * (svv - bvv) * (svv - cvv)
        partial = partial[partial > 0.0]  # to avoid sqrt of negative number
        area = np.sqrt(partial)

        if area.size > 0:
            # radius of circumcircle
            circum_r = avv * bvv * cvv / (4.0 * area)

            # if radius less then alpha then add outer edge
            if circum_r < alpha or alpha == 0:
                add_edge(edges, ia, ib)
                add_edge(edges, ib, ic)
                add_edge(edges, ic, ia)

    return edges


def _get_polygon_coords_from_edges_shapely(
    coords: np.ndarray, edges: set[tuple[int, int]]
) -> list[shapely.coords.CoordinateSequence]:
    """Split and connect edges into shapely polygons"""
    logger.info("Getting polygon coordinates using shapely")
    lines = sg.MultiLineString([(coords[e[0]], coords[e[1]]) for e in edges])
    # Need to use node workaround here to handle crossing lines
    # see https://github.com/shapely/shapely/issues/1736
    noded = shapely.node(shapely.GeometryCollection(lines))
    polygons = polygonize(noded.geoms)
    return [pol.exterior.coords for pol in polygons]


def simplify_polygons(self, tolerance: float, preserve_topology: bool) -> bool:
    """Use Shapely's 'simplify' method to reduce points.

    Note that attributes are not yet supported (perhaps impossible?)

    For Args, see Shapely
    """
    try:
        if self.attributes:
            raise UserWarning(
                "Attributes are present, but they will be lost when simplifying"
            )
    except AttributeError:
        pass

    recompute_hlen = self.hname in self.get_dataframe(copy=False)
    recompute_tlen = self.tname in self.get_dataframe(copy=False)

    orig_len = len(self.get_dataframe(copy=False))

    idgroups = self.get_dataframe(copy=False).groupby(self.pname)
    dfrlist = []
    for idx, grp in idgroups:
        if len(grp.index) < 2:
            logger.warning("Cannot simplify polygons with less than two points. Skip")
            continue

        pxcor = grp[self.xname].values
        pycor = grp[self.yname].values
        pzcor = grp[self.zname].values
        spoly = sg.LineString(np.stack([pxcor, pycor, pzcor], axis=1))

        new_spoly = spoly.simplify(tolerance, preserve_topology=preserve_topology)
        dfr = pd.DataFrame(
            np.array(new_spoly.coords), columns=[self.xname, self.yname, self.zname]
        )

        dfr[self.pname] = idx
        dfrlist.append(dfr)

    dfr = pd.concat(dfrlist)
    self.set_dataframe(dfr.reset_index(drop=True))

    if recompute_hlen:
        self.hlen()
    if recompute_tlen:
        self.tlen()

    new_len = len(self.get_dataframe(copy=False))

    return new_len < orig_len
