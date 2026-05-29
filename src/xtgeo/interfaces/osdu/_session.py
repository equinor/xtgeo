# -*- coding: utf-8 -*-
"""OSDU/RDDMS session configuration and token management.

Handles:
  - Connection parameters (ETP URL, REST base URL)
  - OAuth2 token refresh (refresh_token or client_credentials grant)
  - OSDU access control defaults (ACL owners/viewers, legal tags, countries)
  - Dataspace identity
  - Persistence to ~/.config/xtgeo/osdu/<profile>.toml

Environment variable overrides (prefix XTGEO_OSDU_):
  XTGEO_OSDU_URL, XTGEO_OSDU_TOKEN_URL, XTGEO_OSDU_CLIENT_ID,
  XTGEO_OSDU_CLIENT_SECRET, XTGEO_OSDU_REFRESH_TOKEN,
  XTGEO_OSDU_DATA_PARTITION, XTGEO_OSDU_DATASPACE,
  XTGEO_OSDU_LEGAL_TAG, XTGEO_OSDU_OWNERS, XTGEO_OSDU_VIEWERS,
  XTGEO_OSDU_COUNTRIES

Usage::

    from xtgeo.interfaces.osdu import OsduSession

    # Load or create session
    session = OsduSession.load("my-cloud")
    # or from env/defaults
    session = OsduSession.from_env()

    # Get a live access token (auto-refreshes)
    token = session.access_token()

    # Use with EtpProvider
    from xtgeo.interfaces.osdu import EtpProvider
    provider = EtpProvider(session.etp_config())
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_ENV_PREFIX = "XTGEO_OSDU_"
_CONFIG_DIR = Path.home() / ".config" / "xtgeo" / "osdu"


@dataclass
class OsduSession:
    """OSDU/RDDMS session configuration.

    Parameters
    ----------
    profile : str
        Session profile name (used for file persistence).
    etp_url : str
        WebSocket URL for ETP 1.2 (e.g. ws://localhost:9002 or
        wss://host/api/reservoir-ddms-etp/v2/).
    rest_base_url : str
        REST API base (e.g. https://host.energy.azure.com). Used for
        dataspace creation via OSDU REST API.
    token_url : str
        OAuth2 token endpoint (e.g.
        https://login.microsoftonline.com/<tenant>/oauth2/v2.0/token).
    client_id : str
        Azure AD app registration client ID.
    client_secret : str
        Client secret (for client_credentials or confidential refresh).
    refresh_token : str
        OAuth2 refresh token for token renewal.
    scope : str
        OAuth2 scope (defaults to <client_id>/.default openid offline_access).
    auth_mode : str
        One of: "refresh_token", "client_credentials", "none".
        "none" means no token management (for local dev servers).
    data_partition : str
        OSDU data partition ID (e.g. "opendes", "my-partition").
    dataspace : str
        ETP dataspace path (two-part string, e.g. "myteam/project").
    legal_tag : str
        Default OSDU legal tag for new objects.
    owners : list[str]
        Default ACL owner groups.
    viewers : list[str]
        Default ACL viewer groups.
    countries : list[str]
        Legal countries (ISO 2-letter codes).
    timeout_s : float
        Connection/request timeout in seconds.
    """

    profile: str = "default"

    # Connection
    etp_url: str = "ws://localhost:9002"
    rest_base_url: str = ""
    data_partition: str = ""
    dataspace: str = "xtgeo/default"

    # Auth
    token_url: str = ""
    client_id: str = ""
    client_secret: str = ""
    refresh_token: str = ""
    scope: str = ""
    auth_mode: str = "none"  # "refresh_token" | "client_credentials" | "none"

    # OSDU ACL / Legal defaults
    legal_tag: str = ""
    owners: List[str] = field(default_factory=list)
    viewers: List[str] = field(default_factory=list)
    countries: List[str] = field(default_factory=lambda: ["NO"])

    # Misc
    timeout_s: float = 30.0

    # Runtime (not persisted)
    _access_token: Optional[str] = field(default=None, repr=False)
    _token_expires_at: float = field(default=0.0, repr=False)

    # ------------------------------------------------------------------
    # Token management
    # ------------------------------------------------------------------

    def access_token(self) -> str:
        """Get a valid access token, refreshing if needed.

        Returns empty string if auth_mode is "none" (local dev).
        """
        if self.auth_mode == "none":
            return ""

        # Return cached if still valid (60s margin)
        if self._access_token and time.time() < (self._token_expires_at - 60):
            return self._access_token

        self._refresh()
        return self._access_token or ""

    def _refresh(self) -> None:
        """Refresh the access token via OAuth2."""
        import json
        import urllib.parse
        import urllib.request

        if not self.token_url:
            raise ValueError("token_url required for token refresh")

        if self.auth_mode == "refresh_token":
            if not self.refresh_token:
                raise ValueError("refresh_token required for refresh_token grant")
            scope = self.scope or f"{self.client_id}/.default openid offline_access"
            data = {
                "grant_type": "refresh_token",
                "client_id": self.client_id,
                "refresh_token": self.refresh_token,
                "scope": scope,
            }
            if self.client_secret:
                data["client_secret"] = self.client_secret

        elif self.auth_mode == "client_credentials":
            if not self.client_secret:
                raise ValueError("client_secret required for client_credentials grant")
            scope = self.scope or f"{self.client_id}/.default"
            data = {
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "scope": scope,
            }
        else:
            raise ValueError(f"Unknown auth_mode: {self.auth_mode}")

        encoded = urllib.parse.urlencode(data).encode("utf-8")
        req = urllib.request.Request(
            self.token_url,
            data=encoded,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            body = json.loads(resp.read())

        self._access_token = body["access_token"]
        self._token_expires_at = time.time() + body.get("expires_in", 3600)

        # Rotate refresh token if a new one was issued
        if "refresh_token" in body and body["refresh_token"] != self.refresh_token:
            self.refresh_token = body["refresh_token"]
            logger.debug("Refresh token rotated")

    # ------------------------------------------------------------------
    # ETP config bridge
    # ------------------------------------------------------------------

    def etp_config(self):
        """Create an EtpConnectionConfig from this session."""
        from ._etp_provider import EtpConnectionConfig

        ds_uri = f"eml:///dataspace('{self.dataspace}')"
        return EtpConnectionConfig(
            url=self.etp_url,
            token=self.access_token(),
            dataspace=ds_uri,
            data_partition=self.data_partition,
            timeout_s=self.timeout_s,
        )

    # ------------------------------------------------------------------
    # Dataspace creation (OSDU REST API)
    # ------------------------------------------------------------------

    def create_dataspace_rest(self, path: Optional[str] = None) -> Dict[str, Any]:
        """Create a dataspace via the OSDU Reservoir DDMS REST API.

        Parameters
        ----------
        path : str, optional
            Dataspace path (e.g. "myteam/project"). Defaults to self.dataspace.

        Returns
        -------
        dict
            Server response.
        """
        import json
        import urllib.request

        if not self.rest_base_url:
            raise ValueError("rest_base_url required for REST dataspace creation")

        path = path or self.dataspace
        token = self.access_token()

        custom_data = {
            "legaltags": [self.legal_tag] if self.legal_tag else [],
            "otherRelevantDataCountries": self.countries,
            "viewers": self.viewers,
            "owners": self.owners,
        }

        payload = [{"DataspaceId": path, "Path": path, "CustomData": custom_data}]

        url = f"{self.rest_base_url.rstrip('/')}/api/reservoir-ddms/v2/dataspaces"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "data-partition-id": self.data_partition,
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read())

    def create_dataspace_etp(self, path: Optional[str] = None) -> None:
        """Create a dataspace via ETP PutDataspaces protocol.

        Works for local RDDMS servers that don't require REST API.
        """
        from ._etp_provider import EtpProvider

        path = path or self.dataspace
        config = self.etp_config()
        provider = EtpProvider(config)
        provider.open()

        try:
            import time as _time

            from energistics.etp.v12.datatypes.object.dataspace import Dataspace
            from energistics.etp.v12.protocol.dataspace import PutDataspaces

            now_us = int(_time.time() * 1_000_000)
            ds_uri = f"eml:///dataspace('{path}')"
            ds = Dataspace(
                uri=ds_uri,
                path=path,
                store_last_write=now_us,
                store_created=now_us,
                custom_data={},
            )

            async def _put():
                msg = PutDataspaces(dataspaces={"0": ds})
                responses = await provider._send_and_recv(msg)
                for r in responses:
                    if hasattr(r, "error") and r.error:
                        raise RuntimeError(f"PutDataspaces failed: {r.error}")
                    if hasattr(r, "errors") and r.errors:
                        for k, e in r.errors.items():
                            raise RuntimeError(
                                f"PutDataspaces error [{k}]: {e.message}"
                            )

            provider._run(_put())
            logger.info("Dataspace '%s' created via ETP", path)
        finally:
            provider.close()

    # ------------------------------------------------------------------
    # REST Administration (Dataspace & Object Management)
    # ------------------------------------------------------------------

    def _rest_request(
        self,
        method: str,
        endpoint: str,
        *,
        payload: Any = None,
        params: Optional[Dict[str, str]] = None,
    ) -> Any:
        """Make an authenticated REST API request.

        Parameters
        ----------
        method : str
            HTTP method (GET, POST, PUT, DELETE).
        endpoint : str
            API endpoint path (appended to rest_base_url).
        payload : any, optional
            JSON-serializable request body.
        params : dict, optional
            URL query parameters.

        Returns
        -------
        any
            Parsed JSON response (or None for 204).
        """
        import json
        import urllib.parse
        import urllib.request

        if not self.rest_base_url:
            raise ValueError("rest_base_url required for REST operations")

        base = self.rest_base_url.rstrip("/")
        url = f"{base}/{endpoint.lstrip('/')}"

        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"

        token = self.access_token()
        headers = {
            "Content-Type": "application/json",
            "data-partition-id": self.data_partition,
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"

        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(url, data=data, headers=headers, method=method)

        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            body = resp.read()
            if not body:
                return None
            return json.loads(body)

    def list_dataspaces(self) -> List[Dict[str, Any]]:
        """List all dataspaces accessible via REST API.

        Returns
        -------
        list of dict
            Each dict has keys like 'DataspaceId', 'Path', etc.

        Examples
        --------
        >>> session = OsduSession.load("my-cloud")
        >>> for ds in session.list_dataspaces():
        ...     print(ds['Path'])
        """
        return self._rest_request("GET", "api/reservoir-ddms/v2/dataspaces")

    def get_dataspace(self, path: Optional[str] = None) -> Dict[str, Any]:
        """Get metadata for a specific dataspace.

        Parameters
        ----------
        path : str, optional
            Dataspace path. Defaults to session's dataspace.

        Returns
        -------
        dict
            Dataspace metadata.
        """
        path = path or self.dataspace
        return self._rest_request("GET", f"api/reservoir-ddms/v2/dataspaces/{path}")

    def delete_dataspace(self, path: Optional[str] = None) -> None:
        """Delete a dataspace via REST API.

        Parameters
        ----------
        path : str, optional
            Dataspace path to delete. Defaults to session's dataspace.

        .. warning::
            This permanently removes all data in the dataspace.
        """
        path = path or self.dataspace
        self._rest_request("DELETE", f"api/reservoir-ddms/v2/dataspaces/{path}")
        logger.info("Dataspace '%s' deleted via REST", path)

    def list_objects_rest(
        self,
        dataspace: Optional[str] = None,
        object_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List RESQML objects in a dataspace via REST API.

        Parameters
        ----------
        dataspace : str, optional
            Dataspace path. Defaults to session's dataspace.
        object_type : str, optional
            Filter by RESQML type (e.g. "IjkGrid", "Grid2d").

        Returns
        -------
        list of dict
            Objects with 'uuid', 'title', 'type', 'dataObjectType' keys.
        """
        ds = dataspace or self.dataspace
        params: Dict[str, str] = {"dataspace": ds}
        if object_type:
            params["type"] = object_type
        return self._rest_request("GET", "api/reservoir-ddms/v2/objects", params=params)

    def search_objects_rest(
        self,
        query: str,
        *,
        dataspace: Optional[str] = None,
        object_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Search for objects by title/keyword via REST API.

        Parameters
        ----------
        query : str
            Search query string (matches against title and metadata).
        dataspace : str, optional
            Dataspace to search in. Defaults to session's dataspace.
        object_type : str, optional
            Filter by RESQML type.
        limit : int
            Maximum results to return.

        Returns
        -------
        list of dict
            Matching objects.

        Examples
        --------
        >>> results = session.search_objects_rest("PORO", object_type="Property")
        >>> results = session.search_objects_rest("Drogon")
        """
        ds = dataspace or self.dataspace
        payload: Dict[str, Any] = {
            "query": query,
            "dataspace": ds,
            "limit": limit,
        }
        if object_type:
            payload["type"] = object_type
        return self._rest_request(
            "POST", "api/reservoir-ddms/v2/objects/search", payload=payload
        )

    def get_object_metadata_rest(
        self,
        uuid: str,
        dataspace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get metadata for a specific RESQML object via REST.

        Parameters
        ----------
        uuid : str
            Object UUID.
        dataspace : str, optional
            Dataspace containing the object.

        Returns
        -------
        dict
            Full object metadata.
        """
        ds = dataspace or self.dataspace
        return self._rest_request(
            "GET",
            f"api/reservoir-ddms/v2/objects/{uuid}",
            params={"dataspace": ds},
        )

    def switch_dataspace(self, path: str) -> None:
        """Switch this session to a different dataspace.

        Parameters
        ----------
        path : str
            New dataspace path (e.g. "myteam/production").
        """
        self.dataspace = path
        logger.info("Switched to dataspace '%s'", path)

    # ------------------------------------------------------------------
    # Persistence (TOML)
    # ------------------------------------------------------------------

    def save(self, path: Optional[Path] = None) -> Path:
        """Save session config to TOML file.

        Parameters
        ----------
        path : Path, optional
            Custom file path. Defaults to ~/.config/xtgeo/osdu/<profile>.toml

        Returns
        -------
        Path
            The file path written to.
        """
        if path is None:
            path = _CONFIG_DIR / f"{self.profile}.toml"

        path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            f"# xtgeo OSDU session: {self.profile}",
            f'profile = "{self.profile}"',
            "",
            "[connection]",
            f'etp_url = "{self.etp_url}"',
            f'rest_base_url = "{self.rest_base_url}"',
            f'data_partition = "{self.data_partition}"',
            f'dataspace = "{self.dataspace}"',
            f"timeout_s = {self.timeout_s}",
            "",
            "[auth]",
            f'token_url = "{self.token_url}"',
            f'client_id = "{self.client_id}"',
            "# client_secret and refresh_token should come from environment",
            "# XTGEO_OSDU_CLIENT_SECRET and XTGEO_OSDU_REFRESH_TOKEN",
            f'auth_mode = "{self.auth_mode}"',
            f'scope = "{self.scope}"',
            "",
            "[osdu]",
            f'legal_tag = "{self.legal_tag}"',
            f"owners = {_toml_list(self.owners)}",
            f"viewers = {_toml_list(self.viewers)}",
            f"countries = {_toml_list(self.countries)}",
        ]

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("Session saved to %s", path)
        return path

    @classmethod
    def load(
        cls, profile: str = "default", path: Optional[Path] = None
    ) -> "OsduSession":
        """Load session from TOML file, with env overrides.

        Parameters
        ----------
        profile : str
            Profile name to load.
        path : Path, optional
            Custom file path. Defaults to ~/.config/xtgeo/osdu/<profile>.toml
        """
        if path is None:
            path = _CONFIG_DIR / f"{profile}.toml"

        data: Dict[str, Any] = {}
        if path.exists():
            data = _parse_toml(path)
        else:
            logger.debug("No config file at %s, using defaults + env", path)

        conn = data.get("connection", {})
        auth = data.get("auth", {})
        osdu = data.get("osdu", {})

        session = cls(
            profile=data.get("profile", profile),
            etp_url=conn.get("etp_url", "ws://localhost:9002"),
            rest_base_url=conn.get("rest_base_url", ""),
            data_partition=conn.get("data_partition", ""),
            dataspace=conn.get("dataspace", "xtgeo/default"),
            timeout_s=conn.get("timeout_s", 30.0),
            token_url=auth.get("token_url", ""),
            client_id=auth.get("client_id", ""),
            client_secret=auth.get("client_secret", ""),
            refresh_token=auth.get("refresh_token", ""),
            scope=auth.get("scope", ""),
            auth_mode=auth.get("auth_mode", "none"),
            legal_tag=osdu.get("legal_tag", ""),
            owners=osdu.get("owners", []),
            viewers=osdu.get("viewers", []),
            countries=osdu.get("countries", ["NO"]),
        )

        # Environment overrides (secrets should always come from env)
        session._apply_env_overrides()
        return session

    @classmethod
    def from_env(cls) -> "OsduSession":
        """Create session entirely from environment variables."""
        session = cls()
        session._apply_env_overrides()
        return session

    def _apply_env_overrides(self) -> None:
        """Override fields from XTGEO_OSDU_* environment variables.

        Also supports common OSDU env var patterns as fallback:
          - OSDU_HOSTNAME → derives etp_url and rest_base_url
          - OSDU_TENANT_ID → derives token_url
          - OSDU_CLIENT_ID, OSDU_CLIENT_SECRET, OSDU_SCOPE
          - OSDU_DATA_PARTITION, OSDU_DATASPACE
          - OSDU_LEGAL_TAG, OSDU_ACL_OWNERS, OSDU_ACL_VIEWERS, OSDU_COUNTRIES
          - INSTANCE_<NAME>_* (ores-style) as lowest priority
        """
        _map = {
            "URL": "etp_url",
            "ETP_URL": "etp_url",
            "REST_BASE_URL": "rest_base_url",
            "TOKEN_URL": "token_url",
            "CLIENT_ID": "client_id",
            "CLIENT_SECRET": "client_secret",
            "REFRESH_TOKEN": "refresh_token",
            "SCOPE": "scope",
            "AUTH_MODE": "auth_mode",
            "DATA_PARTITION": "data_partition",
            "DATASPACE": "dataspace",
            "LEGAL_TAG": "legal_tag",
            "TIMEOUT": "timeout_s",
        }

        for env_key, attr in _map.items():
            val = os.environ.get(f"{_ENV_PREFIX}{env_key}")
            if val:
                if attr == "timeout_s":
                    setattr(self, attr, float(val))
                else:
                    setattr(self, attr, val)

        # List fields (comma-separated)
        for env_key, attr in [
            ("OWNERS", "owners"),
            ("VIEWERS", "viewers"),
            ("COUNTRIES", "countries"),
        ]:
            val = os.environ.get(f"{_ENV_PREFIX}{env_key}")
            if val:
                setattr(self, attr, [x.strip() for x in val.split(",") if x.strip()])

        # --- Fallback: common OSDU_* env vars ---
        hostname = os.environ.get("OSDU_HOSTNAME", "")
        if hostname and not os.environ.get(f"{_ENV_PREFIX}ETP_URL"):
            if not self.etp_url or self.etp_url == "ws://localhost:9002":
                self.etp_url = f"wss://{hostname}/api/reservoir-ddms-etp/v2/"
            if not self.rest_base_url:
                self.rest_base_url = f"https://{hostname}"

        tenant_id = os.environ.get("OSDU_TENANT_ID", "")
        if tenant_id and not self.token_url:
            self.token_url = (
                f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
            )

        # Simple OSDU_* fallbacks (lower priority than XTGEO_OSDU_*)
        _osdu_fallbacks = {
            "OSDU_CLIENT_ID": "client_id",
            "OSDU_CLIENT_SECRET": "client_secret",
            "OSDU_SCOPE": "scope",
            "OSDU_DATA_PARTITION": "data_partition",
            "OSDU_DATASPACE": "dataspace",
            "OSDU_LEGAL_TAG": "legal_tag",
        }
        _defaults = {"dataspace": "xtgeo/default"}
        for env_key, attr in _osdu_fallbacks.items():
            current = getattr(self, attr)
            if not current or current == _defaults.get(attr):
                val = os.environ.get(env_key, "")
                if val:
                    setattr(self, attr, val)

        # OSDU_ACL_OWNERS / OSDU_ACL_VIEWERS fallback
        if not self.owners:
            val = os.environ.get("OSDU_ACL_OWNERS", "")
            if val:
                self.owners = [x.strip() for x in val.split(",") if x.strip()]
        if not self.viewers:
            val = os.environ.get("OSDU_ACL_VIEWERS", "")
            if val:
                self.viewers = [x.strip() for x in val.split(",") if x.strip()]
        if self.countries == ["NO"]:
            val = os.environ.get("OSDU_COUNTRIES", "")
            if val:
                self.countries = [x.strip() for x in val.split(",") if x.strip()]

        # Auto-detect auth_mode from available credentials
        if self.auth_mode == "none":
            if self.client_secret and self.refresh_token:
                self.auth_mode = "refresh_token"
            elif self.client_secret:
                self.auth_mode = "client_credentials"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @classmethod
    def list_profiles(cls) -> List[str]:
        """List available saved profiles."""
        if not _CONFIG_DIR.exists():
            return []
        return [p.stem for p in _CONFIG_DIR.glob("*.toml")]

    def __repr__(self) -> str:
        return (
            f"OsduSession(profile={self.profile!r}, etp_url={self.etp_url!r}, "
            f"dataspace={self.dataspace!r}, auth_mode={self.auth_mode!r})"
        )


# ------------------------------------------------------------------
# TOML helpers (minimal, no external deps)
# ------------------------------------------------------------------


def _toml_list(items: List[str]) -> str:
    """Format a list as TOML array."""
    return "[" + ", ".join(f'"{x}"' for x in items) + "]"


def _parse_toml(path: Path) -> Dict[str, Any]:
    """Minimal TOML parser for our flat config structure.

    Supports: key = "value", key = number, key = ["a", "b"],
    [section] headers. No nested tables or multiline.
    """
    import re

    data: Dict[str, Any] = {}
    current_section: Optional[Dict[str, Any]] = None

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Section header
        m = re.match(r"^\[(\w+)\]$", line)
        if m:
            section_name = m.group(1)
            data[section_name] = {}
            current_section = data[section_name]
            continue

        # Key = value
        m = re.match(r"^(\w+)\s*=\s*(.+)$", line)
        if not m:
            continue

        key = m.group(1)
        raw_val = m.group(2).strip()

        # Parse value
        val: Any
        if raw_val.startswith('"') and raw_val.endswith('"'):
            val = raw_val[1:-1]
        elif raw_val.startswith("["):
            # Array of strings
            val = [
                s.strip().strip('"').strip("'")
                for s in raw_val.strip("[]").split(",")
                if s.strip().strip('"').strip("'")
            ]
        elif raw_val.replace(".", "").replace("-", "").isdigit():
            val = float(raw_val) if "." in raw_val else int(raw_val)
        elif raw_val.lower() in ("true", "false"):
            val = raw_val.lower() == "true"
        else:
            val = raw_val

        if current_section is not None:
            current_section[key] = val
        else:
            data[key] = val

    return data
