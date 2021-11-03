import hypothesis.strategies as st

import xtgeo

indices = st.integers(min_value=4, max_value=6)
coordinates = st.floats(min_value=-100.0, max_value=100.0)
increments = st.floats(min_value=1.0, max_value=100.0)
dimensions = st.tuples(indices, indices, indices)

xtgeo_grids = st.builds(
    xtgeo.create_box_grid,
    dimension=dimensions,
    origin=st.tuples(coordinates, coordinates, coordinates),
    increment=st.tuples(increments, increments, increments),
    rotation=st.floats(min_value=0.0, max_value=90),
)
