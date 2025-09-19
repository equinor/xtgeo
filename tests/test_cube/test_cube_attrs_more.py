import pytest

import xtgeo
from tests.conftest import measure_peak_memory_usage
from xtgeo.common.log import functimer


@pytest.mark.bigtest
def test_large_cube_memory_usage():
    """Get attribute around a constant cube slices and check peak memory."""
    import gc

    # Cube size: 500 * 600 * 700 * 4 bytes ~= 840 MB
    mycube = xtgeo.Cube(
        ncol=500, nrow=600, nlay=700, xinc=12, yinc=12, zinc=4, values=666.0
    )

    level1 = 10
    level2 = 1500

    @measure_peak_memory_usage
    @functimer(output="print")
    def compute_with_mem_tracking(cube, interp):
        """Helper function to be decorated."""
        return cube.compute_attributes_in_window(
            level1,
            level2,
            interpolation=interp,
        )

    # Force garbage collection to get a cleaner baseline
    gc.collect()

    for intp in ["linear", "cubic"]:
        print(f"\nTesting interpolation={intp}")
        peak_mem_increase_bytes, result = compute_with_mem_tracking(mycube, intp)

        del result
        gc.collect()  # Clean up before the next run

        peak_mem_increase_gb = peak_mem_increase_bytes / (1024 * 1024 * 1024)

        # The cube itself is ~840MB. The operation needs more.
        # Let's set a reasonable upper bound for this cube size.
        assert peak_mem_increase_gb > 0.001, "Peak memory increase is unexpectedly low."
        assert peak_mem_increase_gb < 1, "Memory usage increase exceeded threshold"
