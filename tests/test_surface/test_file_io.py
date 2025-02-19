import numpy as np
import pytest
from hypothesis import given, strategies as st

import xtgeo
from xtgeo import Cube, RegularSurface


def assert_similar_surfaces(surf1, surf2):
    for attr in ["ncol", "nrow", "xori", "yori", "xinc", "yinc"]:
        assert getattr(surf1, attr) == pytest.approx(
            getattr(surf2, attr), abs=1e-3, rel=1e-3
        )
    assert surf1.values.data.flatten().tolist() == pytest.approx(
        surf2.values.data.flatten().tolist(), abs=1e-3, rel=1e-3
    )


@st.composite
def surfaces(draw):
    ncol = draw(st.integers(min_value=2, max_value=10))
    nrow = draw(st.integers(min_value=2, max_value=10))
    return draw(
        st.builds(
            RegularSurface,
            ncol=st.just(ncol),
            nrow=st.just(nrow),
            values=st.lists(
                st.floats(
                    allow_nan=False,
                    max_value=1e29,
                    min_value=-xtgeo.UNDEF_LIMIT,
                ),
                min_size=ncol * nrow,
                max_size=ncol * nrow,
            ),
            xori=st.floats(
                min_value=-1e8,
                max_value=1e8,
            ),
            yori=st.floats(
                min_value=-1e8,
                max_value=1e8,
            ),
            xinc=st.floats(
                min_value=1e-2,
                max_value=xtgeo.UNDEF_LIMIT,
            ),
            yinc=st.floats(
                min_value=1e-2,
                max_value=xtgeo.UNDEF_LIMIT,
            ),
        )
    )


@pytest.mark.parametrize(
    "fformat",
    [
        "irap_binary",
        "irap_ascii",
        "zmap_ascii",
        "petromod",
        "ijxyz",
        "zmap",
        "zmap_ascii",
        "xtgregsurf",
    ],
)
@pytest.mark.parametrize(
    "input_val, expected_result",
    [
        (1, [[1.0, 1.0], [1.0, 1.0]]),
        ([1, 2, 3, 4], [[1.0, 2.0], [3.0, 4.0]]),
        (np.zeros((2, 2)), [[0.0, 0.0], [0.0, 0.0]]),
        (np.ma.zeros((2, 2)), [[0.0, 0.0], [0.0, 0.0]]),
        (np.ma.ones((2, 2)), [[1.0, 1.0], [1.0, 1.0]]),
    ],
)
def test_simple_io(tmp_path, monkeypatch, input_val, expected_result, fformat):
    monkeypatch.chdir(tmp_path)
    surf = RegularSurface(ncol=2, nrow=2, xinc=2.0, yinc=2.0, values=input_val)
    surf.to_file("my_file", fformat=fformat)
    surf_from_file = RegularSurface._read_file("my_file", fformat=fformat)
    assert_similar_surfaces(surf, surf_from_file)
    assert surf_from_file.values.data.tolist() == expected_result


@pytest.mark.usefixtures("tmp_path_cwd")
@pytest.mark.parametrize(
    "fformat",
    [
        "irap_binary",
        "irap_ascii",
        "zmap",
        "zmap_ascii",
        "petromod",
        "xtgregsurf",
    ],
)
@given(surf=surfaces())
def test_complex_io(surf, fformat):
    if fformat == "petromod":
        pytest.xfail("Several hypotesis failures (4)")
    surf.to_file("my_file", fformat=fformat)
    surf_from_file = RegularSurface._read_file("my_file", fformat=fformat)
    assert_similar_surfaces(surf, surf_from_file)


@pytest.mark.usefixtures("tmp_path_cwd")
@given(surfaces())
def test_complex_io_hdf_classmethod(surf):
    surf.to_hdf("my_file")
    surf_from_file = xtgeo.surface_from_file("my_file", fformat="hdf5")
    assert_similar_surfaces(surf, surf_from_file)


@given(
    data=st.floats(
        allow_nan=False, max_value=xtgeo.UNDEF_LIMIT, min_value=-xtgeo.UNDEF_LIMIT
    ),
)
def test_read_cube(data):
    cube_input = {
        "xori": 1.0,
        "yori": 2.0,
        "zori": 3.0,
        "ncol": 3,
        "nrow": 3,
        "nlay": 2,
        "xinc": 20.0,
        "yinc": 25.0,
        "zinc": 2.0,
    }
    cube = Cube(**cube_input)
    surf_from_cube = RegularSurface._read_cube(cube, data)
    assert list(set(surf_from_cube.values.data.flatten().tolist())) == pytest.approx(
        [data]
    )

    cube_input.pop("zinc")
    cube_input.pop("nlay")
    cube_input.pop("zori")
    # Make sure it is properly constructed:
    result = {key: getattr(surf_from_cube, key) for key in cube_input}
    assert result == cube_input


def test_surf_io_few_nodes(tmp_path, larger_surface):
    """Test to_file() with to few active nodes shall warn or raise error."""

    surf = RegularSurface(**larger_surface)

    assert surf.nactive == 38

    surf.values = np.ma.masked_where(surf.values > 0, surf.values)

    assert surf.nactive == 3

    with pytest.warns(UserWarning, match="surfaces with fewer than 4 nodes will not"):
        surf.to_file(tmp_path / "anyfile1.gri")

    with pytest.raises(RuntimeError, match="surfaces with fewer than 4 nodes will not"):
        surf.to_file(tmp_path / "anyfile2.gri", error_if_near_empty=True)
