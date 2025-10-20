"""Shared fixtures for interfaces RMS API tests.

Note that these are only loaded if RMS API is available, and shall be without
any direct dependency on xtgeo.

The project contains dummy data only, solely for simple tests vs RMS API.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pytest

rmsapi = pytest.importorskip(
    "rmsapi", reason="RMS API tests require 'rmsapi'package to be installed"
)


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@pytest.fixture(scope="module")
def rms_project_as_folder_path(tmp_path_factory) -> str:
    """Create a temporary RMS project, return its folder path."""

    project_path = (
        tmp_path_factory.mktemp("rms_project_as_folder")
        / "tmp_project_as_folder.rmsxxx"
    )
    logger.info("Creating temporary RMS project at %s", project_path)

    try:
        project = rmsapi.Project.create(name="Dummy Project as folder path")
    except Exception as e:
        print("Error: %s" % e)
    project.save_as(str(project_path))
    project.close()
    return str(project_path)


@pytest.fixture(scope="function")
def rms_project_as_project(tmp_path_factory) -> Any:
    """Create a temporary RMS project, return RMS magic project variable."""

    project_path = (
        tmp_path_factory.mktemp("rms_project_as_project")
        / "tmp_project_as_project.rmsxxx"
    )
    logger.info("Creating temporary RMS project at %s", project_path)
    try:
        project = rmsapi.Project.create(name="Dummy Project as project")
    except Exception as e:
        print("Error: %s" % e)
    project.save_as(str(project_path))
    yield project
    logger.info("Closing temporary RMS project at %s", project_path)
    project.close()


@pytest.fixture(scope="function")
def rms_regular_surface_project(tmp_path_factory) -> Any:
    """Create a temporary RMS project with regular surfaces for testing."""

    tmp_path = tmp_path_factory.mktemp("rms_regular_surface")  # due to scope 'module'
    project_path = tmp_path / "tmp_project.rmsxxx"

    logger.info("Creating temporary RMS project at %s", project_path)
    try:
        project = rmsapi.Project.create(name="Dummy Project")
    except Exception as e:
        print("Error: %s" % e)
    project.save_as(str(project_path))

    # create a regular surfaces
    surf1 = rmsapi.RegularGrid2D.create(
        x_origin=0.0, y_origin=0.0, i_inc=10.0, j_inc=10.0, ni=11, nj=11, rotation=15.0
    )
    values = np.ma.arange(121).reshape((11, 11)).astype(np.float64)
    surf1.set_values(values)
    surf1.name = "Surface1"

    # store on clipboard
    destination = project.clipboard.create_surface(
        surf1.name, None, rmsapi.VerticalDomain.time
    )
    destination.set_grid(surf1)
    # store on clipboard within a folder
    destination = project.clipboard.create_surface(
        surf1.name + "_folder", ["Folder1"], rmsapi.VerticalDomain.time
    )
    destination.set_grid(surf1)

    # store on general 2d data (will be vertical domain depth as default)
    destination = project.general2d_data.create_surface(surf1.name, None)
    destination.set_grid(surf1)

    # store in trends; folder/category not relevant here; not domain either
    destination = project.trends.surfaces.create(surf1.name)
    destination.set_grid(surf1)

    # store in Horizons: TOP1 and TOP2 are the names
    project.horizons.create("TOP1", rmsapi.HorizonType.interpreted)
    project.horizons.create("TOP2", rmsapi.HorizonType.interpreted)

    project.horizons.representations.create(
        "CategoryDepth", rmsapi.GeometryType.surface, rmsapi.VerticalDomain.depth
    )
    h1 = project.horizons["TOP1"]["CategoryDepth"]
    h1.set_grid(surf1)

    surf1.set_values(surf1.get_values() + 100.0)
    project.horizons.representations.create(
        "CategoryDepth", rmsapi.GeometryType.surface, rmsapi.VerticalDomain.depth
    )
    h2 = project.horizons["TOP2"]["CategoryDepth"]
    h2.set_grid(surf1)

    # setting data with domain time
    surf1.set_values(surf1.get_values() + 99.0)
    project.horizons.representations.create(
        "CategoryTime", rmsapi.GeometryType.surface, rmsapi.VerticalDomain.time
    )
    t = project.horizons["TOP1"]["CategoryTime"]
    t.set_grid(surf1)

    # store in zones
    project.zones.create(
        "ISO_TOP", top=project.horizons["TOP1"], bottom=project.horizons["TOP2"]
    )
    project.zones.representations.create(
        "CategoryZoneDepth", rmsapi.GeometryType.surface, rmsapi.VerticalDomain.depth
    )
    z = project.zones["ISO_TOP"]["CategoryZoneDepth"]
    z.set_grid(surf1)

    # add some points data for tests that try to read a surface from points (-> error)
    project.horizons.representations.create(
        "CategoryPoints", rmsapi.GeometryType.points, rmsapi.VerticalDomain.depth
    )
    p = project.horizons["TOP1"]["CategoryPoints"]
    p.set_values(np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0], [20.0, 20.0, 20.0]]))

    project.save()

    yield project
    project.close()
