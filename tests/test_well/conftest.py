import pytest

from xtgeo import Well


@pytest.fixture()
def string_to_well(setup_tmpdir):
    def wrapper(wellstring, **kwargs):
        """It is currently not possible to initiate from spec.
        We work around by dumping to csv before reloading
        """
        fpath = "well_data.rmswell"
        with open(fpath, "w") as fh:
            fh.write(wellstring)

        well = Well(fpath, **kwargs)

        return well

    yield wrapper
