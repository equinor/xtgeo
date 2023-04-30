import math

import xtgeo


def create_boundary(self, alpha_factor, is_convex, simplify):
    """Create boundary polygons for a surface."""

    # make into points
    points = xtgeo.points_from_surface(self)

    # compute minimum alpha based on surface resolution
    alpha = math.ceil(math.sqrt(self.xinc**2 + self.yinc**2) / 2)

    pol = xtgeo.Polygons.boundary_from_points(points, alpha_factor, alpha, is_convex)

    if pol is None:
        raise RuntimeError("Could not create a valid Polygons instance for boundary.")

    if simplify:
        if isinstance(simplify, bool):
            pol.simplify(tolerance=0.1)
        elif isinstance(simplify, dict) and "tolerance" in simplify:
            pol.simplify(**simplify)
        else:
            raise ValueError("Invalid values for simplify keyword")

    return pol
