import hypothesis.strategies as st
import numpy as np
from hypothesis.extra.numpy import arrays

from xtgeo.grid3d import GridProperty

from .grdecl_grid_generator import finites
from .grid_generator import xtgeo_grids

keywords = st.text(
    min_size=4, max_size=8, alphabet=st.characters(min_codepoint=60, max_codepoint=123)
)
codes = st.integers(min_value=0, max_value=100)


@st.composite
def grid_properties(draw, name=keywords, grid=xtgeo_grids):
    grid = draw(grid)
    dims = grid.dimensions
    is_discrete = draw(st.booleans())
    _name = draw(name)
    if is_discrete:
        num_codes = draw(st.integers(min_value=2, max_value=10))
        code_names = draw(
            st.lists(min_size=num_codes, max_size=num_codes, elements=name)
        )
        code_values = np.array(
            draw(
                st.lists(
                    unique=True, elements=codes, min_size=num_codes, max_size=num_codes
                )
            ),
            dtype=np.int32,
        )
        values = draw(
            arrays(
                shape=dims,
                dtype=np.int32,
                elements=st.sampled_from(code_values),
            )
        )
        gp = GridProperty(
            grid,
            *dims,
            _name,
            discrete=is_discrete,
            codes=dict(zip(code_values, code_names)),
            values=values,
            roxar_dtype=np.uint16,
            grid=grid,
        )
        gp.dtype = np.int32
        return gp

    values = draw(arrays(shape=dims, dtype=np.float64, elements=finites))
    return GridProperty(
        grid,
        *dims,
        _name,
        discrete=is_discrete,
        values=values,
        grid=grid,
        roxar_dtype=np.float32,
    )
