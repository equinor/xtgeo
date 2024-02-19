import pytest
import xtgeo


@pytest.fixture()
def string_to_well(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def wrapper(wellstring, **kwargs):
        """It is currently not possible to initiate from spec.
        We work around by dumping to csv before reloading
        """
        fpath = "well_data.rmswell"
        with open(fpath, "w") as fh:
            fh.write(wellstring)

        return xtgeo.well_from_file(fpath, **kwargs)

    yield wrapper
