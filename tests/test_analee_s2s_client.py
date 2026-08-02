"""Tests for the reference accountants → Analee S2S client (stdlib HTTP)."""

import json
from unittest import mock

import pytest

from integration.accountants.analee_s2s import (
  AnaleeConfig,
  AnaleeProvisioningError,
  AnaleeWorkspaceClient,
)


@pytest.fixture
def config():
  return AnaleeConfig(
    base_url="https://analee.test",
    secret="shared-secret",
    enabled=True,
  )


def test_config_dark_by_default(monkeypatch):
  monkeypatch.delenv("ANALEE_INTEGRATION_ENABLED", raising=False)
  cfg = AnaleeConfig.from_env()
  assert cfg.enabled is False
  assert cfg.ok() is False


def test_open_workspace_url_uses_login_url(config):
  client = AnaleeWorkspaceClient(config)

  def fake_post(path, body):
    if path.endswith("/workspace"):
      return {"created": True, "client_ref": body["client_ref"]}
    assert path.endswith("/login-link")
    return {
      "found": True,
      "login_url": "https://analee.test/workspace/enter?token=abc",
      "url_path": "/workspace/enter?token=abc",
    }

  with mock.patch.object(client, "_post", side_effect=fake_post):
    url = client.open_workspace_url(
      client_ref="acc-1",
      client_name="Mokoena Traders",
    )
  assert url == "https://analee.test/workspace/enter?token=abc"


def test_open_workspace_url_falls_back_to_base_plus_path(config):
  client = AnaleeWorkspaceClient(config)

  def fake_post(path, body):
    if path.endswith("/workspace"):
      return {"created": False}
    return {"found": True, "url_path": "/workspace/enter?token=xyz"}

  with mock.patch.object(client, "_post", side_effect=fake_post):
    url = client.open_workspace_url(client_ref="acc-1", client_name="X")
  assert url == "https://analee.test/workspace/enter?token=xyz"


def test_not_found_after_ensure_raises(config):
  client = AnaleeWorkspaceClient(config)

  with mock.patch.object(client, "_post", side_effect=[
    {"created": True},
    {"found": False},
  ]):
    with pytest.raises(AnaleeProvisioningError, match="not found"):
      client.open_workspace_url(client_ref="acc-1", client_name="X")


def test_http_error_wrapped(config):
    client = AnaleeWorkspaceClient(config)
    with mock.patch.object(
        client,
        "_post",
        side_effect=AnaleeProvisioningError("Analee HTTP 401"),
    ):
        with pytest.raises(AnaleeProvisioningError, match="401"):
            client.ensure_workspace(client_ref="a", client_name="b")
