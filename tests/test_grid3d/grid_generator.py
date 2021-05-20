import hypothesis.strategies as st
from xtgeo.grid3d import Grid

indecies = st.integers(min_value=4, max_value=10)
coordinates = st.floats(min_value=-100.0, max_value=100.0)
increments = st.floats(min_value=1.0, max_value=100.0)
dimensions = st.tuples(indecies, indecies, indecies)


def create_grid(*args, **kwargs):
    grid = Grid()
    grid.create_box(*args, **kwargs)
    return grid


xtgeo_grids = st.builds(
    create_grid,
    dimension=dimensions,
    origin=st.tuples(coordinates, coordinates, coordinates),
    increment=st.tuples(increments, increments, increments),
    rotation=st.floats(min_value=0.0, max_value=90),
)
