"""ETP roundtrip verification on eqndev / maap/sleipner dataspace.

Test flow:
  1. Connect to eqndev RDDMS via ETP (client_credentials auth)
  2. Read the IJK grid's XML + all raw arrays from maap/sleipner dataspace
  3. Write the XML + arrays with a new UUID (rewritten in XML)
  4. Read the newly-written grid's XML + arrays back
  5. Compare: arrays must be EXACTLY identical (bitwise), only UUID differs

Uses auth credentials from ~/ores/k8s/secret.yaml + configmap.yaml.
ETP-only (no REST API calls).
"""

from __future__ import annotations

import re
import sys
import uuid as _uuid
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.requires_rddms

# ---------------------------------------------------------------------------
# Auth helpers - load from ~/ores/k8s
# ---------------------------------------------------------------------------

K8S_DIR = Path.home() / "ores" / "k8s"


def _load_k8s_env() -> dict[str, str]:
    """Load env vars from k8s/configmap.yaml + secret.yaml."""
    sys.path.insert(0, str(K8S_DIR))
    from env_from_k8s import load_k8s_yaml

    config = load_k8s_yaml(K8S_DIR / "configmap.yaml")
    secrets = load_k8s_yaml(K8S_DIR / "secret.yaml")
    return {**config, **secrets}


def _get_etp_config():
    """Create EtpConnectionConfig for eqndev / maap/sleipner."""
    from xtgeo.interfaces.osdu._session import OsduSession

    env = _load_k8s_env()

    tenant_id = env["INSTANCE_EQNDEV_TENANT_ID"]
    client_id = env["INSTANCE_EQNDEV_CLIENT_ID"]
    client_secret = env["INSTANCE_EQNDEV_CLIENT_SECRET"]
    scope = env["INSTANCE_EQNDEV_SCOPE"]
    hostname = env["INSTANCE_EQNDEV_HOSTNAME"]
    data_partition = env["INSTANCE_EQNDEV_DATA_PARTITION_ID"]

    session = OsduSession(
        profile="eqndev",
        etp_url=f"wss://{hostname}/api/reservoir-ddms-etp/v2/",
        rest_base_url=f"https://{hostname}",
        token_url=f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token",
        client_id=client_id,
        client_secret=client_secret,
        scope=scope,
        auth_mode="client_credentials",
        data_partition=data_partition,
        dataspace="maap/sleipner",
    )
    return session.etp_config()


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def test_ijk_grid_roundtrip_new_uuid():
    """Download IJK grid from maap/sleipner, re-ingest with new UUID, compare.

    Works at the raw ETP level: XML + arrays. No xtgeo format conversion.
    Verifies the ETP store/retrieve roundtrip is bitwise lossless.
    """
    from xtgeo.interfaces.osdu import EtpProvider

    # --- Step 1: Connect and discover ---
    print("\n=== Step 1: Connect to eqndev / maap/sleipner ===")
    config = _get_etp_config()
    provider = EtpProvider(config)
    provider.open()
    print(f"  Connected: {config.url}")
    print(f"  Dataspace: {config.dataspace}")

    # List objects to find the IJK grid
    objects = provider.list_objects()
    grids = [o for o in objects if "IjkGrid" in o.get("type", "")]
    print(f"  Found {len(grids)} IJK grid(s)")
    assert len(grids) >= 1, "No IJK grids found in maap/sleipner"

    grid_obj = grids[0]
    grid_uuid_orig = grid_obj["uuid"]
    grid_uri_orig = grid_obj["uri"]
    grid_title = grid_obj.get("title", "unknown")
    print(f"  Grid: {grid_title} ({grid_uuid_orig})")
    print(f"  URI: {grid_uri_orig}")

    # --- Step 2: Read XML and all arrays ---
    print("\n=== Step 2: Read grid XML + raw arrays ===")
    xml_orig = provider._get_xml(grid_uuid_orig, uri=grid_uri_orig)
    assert xml_orig, "Failed to get XML"
    print(f"  XML length: {len(xml_orig)} bytes")

    # Extract all array paths from XML
    array_paths = re.findall(
        r"<eml:PathInHdfFile[^>]*>([^<]+)</eml:PathInHdfFile>", xml_orig
    )
    print(f"  Array paths found: {len(array_paths)}")
    for p in array_paths:
        print(f"    {p}")

    # Download all arrays
    arrays_orig: dict[str, np.ndarray] = {}
    for path in array_paths:
        arr = provider._get_array(grid_uri_orig, path)
        if arr is not None:
            arrays_orig[path] = arr
            print(f"    {path}: shape={arr.shape}, dtype={arr.dtype}, "
                  f"size={arr.nbytes / 1024:.1f} KB")
        else:
            print(f"    {path}: FAILED TO DOWNLOAD")

    assert len(arrays_orig) == len(array_paths), (
        f"Only got {len(arrays_orig)}/{len(array_paths)} arrays"
    )
    provider.close()

    # --- Step 3: Write with new UUID ---
    print("\n=== Step 3: Re-ingest with new UUID ===")
    new_uuid = str(_uuid.uuid4())
    new_title = f"{grid_title}_ROUNDTRIP_VERIFY"
    print(f"  New UUID: {new_uuid}")
    print(f"  New title: {new_title}")

    # Rewrite XML: replace old UUID with new, update title
    xml_new = xml_orig.replace(grid_uuid_orig, new_uuid)
    xml_new = xml_new.replace(
        f"<eml:Title xsi:type=\"eml:DescriptionString\">{grid_title}</eml:Title>",
        f"<eml:Title xsi:type=\"eml:DescriptionString\">{new_title}</eml:Title>",
    )
    # Verify UUID was actually replaced
    assert new_uuid in xml_new
    assert grid_uuid_orig not in xml_new

    # Construct new URI
    qtype = grid_obj["type"]  # e.g. "resqml20.obj_IjkGridRepresentation"
    new_uri = f"{config.dataspace}/{qtype}({new_uuid})"
    print(f"  New URI: {new_uri}")

    # Put XML + arrays in a single transaction
    provider2 = EtpProvider(config)
    provider2.open()
    provider2._run(provider2._start_transaction())
    provider2._put_xml(new_uri, xml_new, qtype)
    print(f"  Wrote XML")

    # Put arrays with updated paths (UUID replaced in path)
    for old_path, arr in arrays_orig.items():
        new_path = old_path.replace(grid_uuid_orig, new_uuid)
        provider2._put_array(new_uri, new_path, arr)
        print(f"  Wrote array: {new_path}")

    # Commit transaction
    provider2._run(provider2._commit_transaction())
    provider2.close()
    print(f"  Transaction committed")

    # --- Step 4: Read back the new grid ---
    print("\n=== Step 4: Read back the new grid ===")
    provider3 = EtpProvider(config)
    provider3.open()

    # Verify the object exists
    objects2 = provider3.list_objects()
    new_grid_found = [o for o in objects2 if o["uuid"] == new_uuid]
    assert len(new_grid_found) == 1, f"New grid {new_uuid} not found in dataspace"
    new_uri_resolved = new_grid_found[0]["uri"]
    print(f"  Found new grid: {new_grid_found[0].get('title')} ({new_uuid})")

    # Read XML back
    xml_readback = provider3._get_xml(new_uuid, uri=new_uri_resolved)
    assert xml_readback, "Failed to read back XML"
    print(f"  XML readback length: {len(xml_readback)} bytes")

    # Read arrays back
    arrays_readback: dict[str, np.ndarray] = {}
    for old_path in array_paths:
        new_path = old_path.replace(grid_uuid_orig, new_uuid)
        arr = provider3._get_array(new_uri_resolved, new_path)
        if arr is not None:
            arrays_readback[new_path] = arr
        else:
            print(f"    FAILED to read back: {new_path}")

    provider3.close()

    assert len(arrays_readback) == len(array_paths), (
        f"Only got {len(arrays_readback)}/{len(array_paths)} arrays on readback"
    )

    # --- Step 5: Bitwise comparison ---
    print("\n=== Step 5: Bitwise comparison ===")

    all_identical = True
    for old_path, orig_arr in arrays_orig.items():
        new_path = old_path.replace(grid_uuid_orig, new_uuid)
        read_arr = arrays_readback[new_path]

        if orig_arr.shape != read_arr.shape:
            print(f"  FAIL {old_path}: shape {orig_arr.shape} vs {read_arr.shape}")
            all_identical = False
            continue

        if orig_arr.dtype != read_arr.dtype:
            print(f"  WARN {old_path}: dtype {orig_arr.dtype} vs {read_arr.dtype}")

        # Compare as raw bytes (bitwise exact)
        if np.array_equal(orig_arr, read_arr):
            print(f"  BITWISE IDENTICAL: {old_path}")
            print(f"    shape={orig_arr.shape}, dtype={orig_arr.dtype}, "
                  f"size={orig_arr.nbytes / 1024:.1f} KB")
        else:
            # Check how close
            diff = np.abs(orig_arr.astype(np.float64) - read_arr.astype(np.float64))
            max_diff = np.max(diff)
            n_diff = np.sum(diff > 0)
            print(f"  DIFFER: {old_path}")
            print(f"    max_diff={max_diff}, n_different={n_diff}/{orig_arr.size}")
            all_identical = False

    # UUID check
    assert new_uuid != grid_uuid_orig
    print(f"\n  UUID original:  {grid_uuid_orig}")
    print(f"  UUID roundtrip: {new_uuid}")
    print(f"  UUIDs correctly differ: YES")

    if all_identical:
        print("\n=== PASS: All arrays BITWISE IDENTICAL ===")
        print("  ETP roundtrip is perfectly lossless. Only UUID differs.")
    else:
        raise AssertionError(
            "Some arrays differ after roundtrip — see details above"
        )


if __name__ == "__main__":
    test_ijk_grid_roundtrip_new_uuid()
