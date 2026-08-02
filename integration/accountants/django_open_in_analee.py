"""Django wiring example for THE ACCOUNTANTS — copy into ``portal/views.py`` (or similar).

Adjust imports for your Client model and URL names. Keep behind
``ANALEE_INTEGRATION_ENABLED`` until Festus enables Analee provisioning in prod.

Suggested URL (``portal/urls.py``)::

    path(
        "clients/<int:client_id>/analee/",
        views.open_client_in_analee,
        name="open_client_in_analee",
    )

Template snippet (client detail page)::

    {% if analee_integration_enabled %}
      <a class="btn btn-primary"
         href="{% url 'open_client_in_analee' client.pk %}">
        Open in Analee
      </a>
    {% endif %}
"""
from __future__ import annotations

from django.contrib.auth.decorators import login_required
from django.http import HttpResponseRedirect
from django.shortcuts import get_object_or_404
from django.views.decorators.http import require_http_methods

from integration.accountants.analee_s2s import (
  AnaleeConfig,
  AnaleeProvisioningError,
  AnaleeWorkspaceClient,
)


def analee_integration_enabled() -> bool:
  return AnaleeConfig.from_env().ok()


@login_required
@require_http_methods(["GET", "POST"])
def open_client_in_analee(request, client_id: int):
  """Ensure the client's Analee workspace and redirect the accountant's browser."""
  # Replace with your real model import, e.g. from clients.models import Client
  from portal.models import Client  # type: ignore[attr-defined]  # noqa: F401

  client = get_object_or_404(Client, pk=client_id)
  config = AnaleeConfig.from_env()
  if not config.ok():
    # Fail soft while dark — same pattern as Analee's provisioning flag.
    return HttpResponseRedirect(client.get_absolute_url())

  # Stable cross-product id: prefer a dedicated slug/ref field if you have one.
  client_ref = getattr(client, "analee_client_ref", None) or f"client-{client.pk}"
  client_name = str(getattr(client, "display_name", None) or client)
  entity_name = getattr(client, "entity_type_name", None)

  api = AnaleeWorkspaceClient(config)
  try:
    url = api.open_workspace_url(
      client_ref=client_ref,
      client_name=client_name,
      entity_name=entity_name,
    )
  except AnaleeProvisioningError:
    # Log server-side; show a friendly message in your messages framework.
    return HttpResponseRedirect(client.get_absolute_url())

  return HttpResponseRedirect(url)
