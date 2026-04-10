"""Tests for ResInsight rips utility helpers."""

from __future__ import annotations

import pytest

from xtgeo.interfaces.resinsight.rips_utils import RipsApiUtils

pytestmark = pytest.mark.requires_resinsight


def test_init_with_none_auto_discovers(resinsight_instance):
    util = RipsApiUtils(instance_or_port=None)
    assert util.instance.location == resinsight_instance.location, (
        "Should auto-discover the running ResInsight instance"
    )


def test_init_with_port_finds_instance(resinsight_instance):
    port = resinsight_instance.location.split(":")[1]
    util = RipsApiUtils(instance_or_port=int(port))
    assert util.instance.location == resinsight_instance.location, (
        "Should find the instance running on the specified port"
    )


def test_init_with_wrong_port():
    with pytest.raises(
        RuntimeError, match="Unable to connect to a ResInsight instance on port 1234"
    ):
        RipsApiUtils.find_instance(port=1234)  # Assuming this port is not used


def test_init_with_existing_instance(resinsight_instance):
    util = RipsApiUtils(instance_or_port=resinsight_instance)
    assert util.instance == resinsight_instance, (
        "Should use the provided ResInsight instance"
    )


def test_init_with_invalid_type_raises():
    with pytest.raises(TypeError, match="instance_or_port must be"):
        RipsApiUtils(instance_or_port="not_a_valid_type")


# --- instance and project properties ---
def test_instance_property(resinsight_instance):
    util = RipsApiUtils(instance_or_port=resinsight_instance)
    assert util.instance is resinsight_instance


def test_project_property(resinsight_instance):
    util = RipsApiUtils(instance_or_port=resinsight_instance)
    assert util.project is resinsight_instance.project


def test_save_project(resinsight_instance, tmp_path):
    util = RipsApiUtils(instance_or_port=resinsight_instance)
    temp_project_path = tmp_path / "test_project.rips"
    util.save_project(name=str(temp_project_path))
    assert temp_project_path.exists(), (
        "Project file should be created at the specified path"
    )


def test_launch_instance_with_executable():
    instance = RipsApiUtils.launch_instance(executable="", console_mode=True)
    assert instance is not None, "Should launch an instance"
    instance.exit()  # Clean up
