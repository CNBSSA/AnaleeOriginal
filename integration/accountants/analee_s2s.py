"""Reference S2S client for CNBSSA/accountants → Analee client workspaces.

Copy this module into THE ACCOUNTANTS (e.g. ``portal/analee/`` or
``services/analee_provisioning.py``). It mirrors the sealed Analee seam in
``provisioning.py`` — no Analee imports required on the accountants side.

Environment (accountants + Analee must share the bearer secret when live):
    ANALEE_BASE_URL              e.g. https://analee.example (no trailing slash)
    ANALEE_PROVISIONING_SECRET   same value as on Analee
    ANALEE_INTEGRATION_ENABLED   default off; when false, callers should no-op / hide UI
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping, Optional


class AnaleeProvisioningError(Exception):
    """Analee returned a non-success HTTP status or malformed JSON."""


@dataclass(frozen=True)
class AnaleeConfig:
    base_url: str
    secret: str
    enabled: bool = False

    @classmethod
    def from_env(cls) -> "AnaleeConfig":
    enabled = os.environ.get("ANALEE_INTEGRATION_ENABLED", "False") == "True"
    base = (os.environ.get("ANALEE_BASE_URL") or "").strip().rstrip("/")
    secret = os.environ.get("ANALEE_PROVISIONING_SECRET", "") or ""
    return cls(base_url=base, secret=secret, enabled=enabled)

    def ok(self) -> bool:
    return self.enabled and bool(self.base_url and self.secret)


class AnaleeWorkspaceClient:
    """Server-to-server calls into Analee's dark provisioning surface."""

    ENSURE_PATH = "/api/provisioning/analee/workspace"
    LOGIN_LINK_PATH = "/api/provisioning/analee/workspace/login-link"

    def __init__(self, config: AnaleeConfig, timeout_seconds: float = 15.0):
    self._config = config
    self._timeout = timeout_seconds

    def _post(self, path: str, body: Mapping[str, Any]) -> dict:
    if not self._config.ok():
      raise AnaleeProvisioningError("Analee integration is not configured")
    url = f"{self._config.base_url}{path}"
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
      url,
      data=data,
      method="POST",
      headers={
        "Authorization": f"Bearer {self._config.secret}",
        "Content-Type": "application/json",
        "Accept": "application/json",
      },
    )
    try:
      with urllib.request.urlopen(req, timeout=self._timeout) as resp:
        raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
      detail = exc.read().decode("utf-8", errors="replace")
      raise AnaleeProvisioningError(
        f"Analee HTTP {exc.code} for {path}: {detail}"
      ) from exc
    except urllib.error.URLError as exc:
      raise AnaleeProvisioningError(f"Analee request failed: {exc}") from exc
    try:
      return json.loads(raw) if raw else {}
    except json.JSONDecodeError as exc:
      raise AnaleeProvisioningError("Analee returned non-JSON body") from exc

    def ensure_workspace(
    self,
    *,
    client_ref: str,
    client_name: str,
    entity_name: Optional[str] = None,
    ) -> dict:
    """Idempotent workspace ensure; raises on HTTP/transport errors."""
    payload: dict[str, Any] = {
      "client_ref": client_ref,
      "client_name": client_name,
    }
    if entity_name:
      payload["entity_name"] = entity_name
    return self._post(self.ENSURE_PATH, payload)

    def login_link(
    self,
    *,
    client_ref: Optional[str] = None,
    email: Optional[str] = None,
    ) -> dict:
    """Mint a short-TTL login path. Prefer ``client_ref`` (accountants-owned id)."""
    payload: dict[str, Any] = {}
    if client_ref:
      payload["client_ref"] = client_ref
    if email:
      payload["email"] = email
    return self._post(self.LOGIN_LINK_PATH, payload)

    def open_workspace_url(
    self,
    *,
    client_ref: str,
    client_name: str,
    entity_name: Optional[str] = None,
    ) -> str:
    """Ensure workspace exists, then return the browser URL for redirect."""
    self.ensure_workspace(
      client_ref=client_ref,
      client_name=client_name,
      entity_name=entity_name,
    )
    link = self.login_link(client_ref=client_ref)
    if not link.get("found"):
      raise AnaleeProvisioningError("Analee workspace not found after ensure")
    login_url = link.get("login_url")
    if login_url:
      return login_url
    base = self._config.base_url.rstrip("/")
    url_path = link.get("url_path") or ""
    if not url_path.startswith("/"):
      raise AnaleeProvisioningError("Analee login-link missing url_path")
    return f"{base}{url_path}"
