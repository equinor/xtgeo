import pytest
import deprecation

from hypothesis import given, settings
from hypothesis import strategies as st
import numpy as np

from xtgeo import RegularSurface, Cube
import xtgeo


def assert_equal_to_init(init, result):
    init = init.copy()
    init.pop("values")
    result_dict = {key: getattr(result, key) for key in init.keys()}
    assert result_dict == pytest.approx(init, abs=1e-3, rel=1e-3)


@st.composite
def generate_data(draw):
    base_data = st.fixed_dictionaries(
        {
            "ncol": st.integers(min_value=2, max_value=10),
            "nrow": st.integers(min_value=2, max_value=10),
            "xori": st.floats(
                min_value=-1e8,
                max_value=1e8,
            ),
            "yori": st.floats(
                min_value=-1e8,
                max_value=1e8,
            ),
            "xinc": st.floats(
                min_value=1e-2,
                max_value=xtgeo.UNDEF_LIMIT,
            ),
            "yinc": st.floats(
                min_value=1e-2,
                max_value=xtgeo.UNDEF_LIMIT,
            ),
        }
    )
    base = draw(base_data)
    values = st.fixed_dictionaries(
        {
            "values": st.lists(
                st.floats(
                    allow_nan=False,
                    max_value=1e29,
                    min_value=-xtgeo.UNDEF_LIMIT,
                ),
                min_size=base["ncol"] * base["nrow"],
                max_size=base["ncol"] * base["nrow"],
            )
        }
    )
    vals = draw(values)

    return {**base, **vals}


@pytest.mark.usefixtures("setup_tmpdir")
@pytest.mark.parametrize("engine", ["cxtgeo", "python"])
@pytest.mark.parametrize(
    "fformat",
    [
        "irap_binary",
        "irap_ascii",
        "zmap_ascii",
        # "ijxyz",  # This fails horribly
        "petromod",
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
def test_simple_io(input_val, expected_result, fformat, engine):
    if engine == "python" and fformat not in [
        "irap_ascii",
        "irap_binary",
        "zmap_ascii",
    ]:
        pytest.skip("Only one engine available")
    init_dict = {"ncol": 2, "nrow": 2, "xinc": 2.0, "yinc": 2.0, "values": input_val}
    surf = RegularSurface(**init_dict)
    surf.to_file("my_file", fformat=fformat)
    surf_from_file = RegularSurface._read_file(
        "my_file", fformat=fformat, engine=engine
    )
    assert_equal_to_init(init_dict, surf_from_file)
    assert surf_from_file.values.data.tolist() == expected_result


@settings(deadline=None)
@pytest.mark.usefixtures("setup_tmpdir")
@pytest.mark.parametrize("input_engine", ["cxtgeo", "python"])
@pytest.mark.parametrize("output_engine", ["cxtgeo", "python"])
@pytest.mark.parametrize(
    "fformat",
    [
        "irap_binary",
        "irap_ascii",
        "zmap",
        "zmap_ascii",
        # "ijxyz",  # This fails horribly
        "petromod",
        "xtgregsurf",
    ],
)
@given(data=generate_data())
def test_complex_io(data, fformat, output_engine, input_engine):
    if (input_engine == "python" or output_engine == "python") and fformat not in [
        "irap_ascii",
        "irap_binary",
        "zmap_ascii",
    ]:
        pytest.skip("Only one engine available")
    if fformat == "petromod":
        pytest.xfail("Several hypotesis failures (4)")
    surf = RegularSurface(**data)
    assert_equal_to_init(data, surf)
    surf.to_file("my_file", fformat=fformat, engine=output_engine)
    surf_from_file = RegularSurface._read_file(
        "my_file", fformat=fformat, engine=input_engine
    )
    assert_equal_to_init(data, surf_from_file)
    assert surf_from_file.values.data.flatten().tolist() == pytest.approx(
        data["values"]
    )


@deprecation.fail_if_not_removed
@pytest.mark.usefixtures("setup_tmpdir")
@settings(deadline=400)
@given(
    data=st.lists(
        st.floats(
            allow_nan=False, max_value=xtgeo.UNDEF_LIMIT, min_value=-xtgeo.UNDEF_LIMIT
        ),
        min_size=9,
        max_size=9,
    )
)
def test_complex_io_hdf(data):
    surf = RegularSurface(3, 3, 0.0, 0.0, values=data)
    surf.to_hdf("my_file")
    surf_from_file = RegularSurface(1, 1, 0.0, 0.0)  # <- unimportant as it gets reset
    surf_from_file.from_hdf("my_file")
    assert surf_from_file.values.data.flatten().tolist() == pytest.approx(data)


@pytest.mark.usefixtures("setup_tmpdir")
@given(
    data=st.lists(
        st.floats(
            allow_nan=False, max_value=xtgeo.UNDEF_LIMIT, min_value=-xtgeo.UNDEF_LIMIT
        ),
        min_size=9,
        max_size=9,
    )
)
def test_complex_io_hdf_classmethod(data):
    surf = RegularSurface(3, 3, 0.0, 0.0, values=data)
    surf.to_hdf("my_file")
    surf_from_file = xtgeo.surface_from_file("my_file", fformat="hdf5")
    assert surf_from_file.values.data.flatten().tolist() == pytest.approx(data)


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
    assert set(surf_from_cube.values.data.flatten().tolist()) == pytest.approx({data})

    cube_input.pop("zinc")
    cube_input.pop("nlay")
    cube_input.pop("zori")
    # Make sure it is properly constructed:
    result = {key: getattr(surf_from_cube, key) for key in cube_input}
    assert result == cube_input
