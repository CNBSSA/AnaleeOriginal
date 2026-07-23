# The Accountants ↔ Analee — workspace provisioning contract

**Audience:** `CNBSSA/accountants` implementers and cloud agents.  
**Analee source:** `provisioning.py` on branch `cursor/cnbssa-accountants-integration-a0c1` (target: merge to `develop`).  
**Status:** Dark until Analee sets `ANALEE_PROVISIONING_ENABLED=True` and a shared secret.

Analee is frozen; this seam is **additive** and **server-to-server only**. THE ACCOUNTANTS owns client mapping, UI, and TB import into `dataimports/` / portal.

---

## Environment variables

### Analee (Railway / deploy)

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `ANALEE_PROVISIONING_ENABLED` | Yes (to go live) | `False` | `True` exposes provisioning routes; `False` → **404** (dark). |
| `ANALEE_PROVISIONING_SECRET` | Yes (when enabled) | — | Bearer token; missing → **503** on S2S routes. |
| `ANALEE_PUBLIC_BASE_URL` | Recommended | — | Public origin, no trailing slash (e.g. `https://analee.example.com`). When set, login-link responses include absolute `login_url`. |
| `ANALEE_WORKSPACE_LINK_TTL` | No | `90` | Seconds for signed workspace login tokens. |

### The Accountants (consumer)

| Variable | Required | Purpose |
|----------|----------|---------|
| `ANALEE_BASE_URL` | Yes | Analee public origin, **no trailing slash** — prefix for all paths below. |
| `ANALEE_PROVISIONING_SECRET` | Yes | **Same value** as on Analee. Sent as `Authorization: Bearer <secret>`. |

Optional feature flag on accountants side (recommended): e.g. `ANALEE_WORKSPACE_ENABLED=False` until staging is wired.

**Never** log the secret or embed it in client-side JavaScript.

---

## Authentication (all three endpoints)

```
Authorization: Bearer <ANALEE_PROVISIONING_SECRET>
Content-Type: application/json
```

| Condition | HTTP |
|-----------|------|
| Provisioning disabled on Analee | 404 `{"error":"not found"}` |
| Secret not configured on Analee | 503 |
| Wrong/missing bearer | 401 |

---

## `client_ref` (idempotency key)

THE ACCOUNTANTS supplies a **stable** identifier per firm client (e.g. internal client PK or UUID string).

Analee sanitises: lowercase, non `[a-z0-9]` → `-`, trim `-`, max 60 chars. Empty after sanitise → **400**.

Workspace email is deterministic:

```
client+<sanitised-ref>@ws.theaccountants.local
```

Use the **same** `client_ref` string on ensure, login-link, and trial-balance. Store `workspace_user_id` / `email` from ensure response if useful for support.

---

## Endpoint 1 — Ensure workspace

**`POST {ANALEE_BASE_URL}/api/provisioning/analee/workspace`**

### Request body

```json
{
  "client_ref": "acc-7-42",
  "client_name": "Mokoena Traders (Pty) Ltd",
  "entity_name": "Private Company"
}
```

- `client_ref` — required (after sanitisation rules).
- `client_name` — display name in Analee `CompanySettings`.
- `entity_name` — optional; must match an Analee `Entity.name` (e.g. `Sole Proprietor`). Unknown → default Private Company chart.

### Success `200`

```json
{
  "created": true,
  "workspace_user_id": 123,
  "email": "client+acc-7-42@ws.theaccountants.local",
  "client_ref": "acc-7-42",
  "company": "Mokoena Traders (Pty) Ltd",
  "entity": "Private Company",
  "chart_provisioned": true
}
```

Re-ensure: `created: false`, same `workspace_user_id`, may refresh `company` name and reactivate subscription.

### Errors

- **400** — bad `client_ref`.
- **401** / **503** / **404** — auth/dark (see above).

---

## Endpoint 2 — Mint login link

**`POST {ANALEE_BASE_URL}/api/provisioning/analee/workspace/login-link`**

### Request body (prefer `client_ref`)

```json
{
  "client_ref": "acc-7-42"
}
```

Alternatively `email` if already stored (must be `*@ws.theaccountants.local`).

### Success `200`

```json
{
  "found": true,
  "client_ref": "acc-7-42",
  "url_path": "/workspace/enter?token=…",
  "expires_in": 90,
  "login_url": "https://analee.example.com/workspace/enter?token=…"
}
```

- `login_url` present only when Analee has `ANALEE_PUBLIC_BASE_URL` set.
- If workspace missing/inactive: `{"found": false}` (still **200**).
- **400** if email is not a workspace alias (blocks human account takeover).

### Browser flow

Redirect the accountant to `login_url` if present, else `ANALEE_BASE_URL + url_path`.  
Analee verifies token → sets `session['workspace_session']` → dashboard.  
Links expire quickly (`expires_in`); re-mint on expiry.

---

## Endpoint 3 — Pull trial balance (S2S)

**`POST {ANALEE_BASE_URL}/api/provisioning/analee/workspace/trial-balance`**

### Request body

```json
{
  "client_ref": "acc-7-42"
}
```

### Success `200` (data ready)

Phase 5 JSON (same contract as authenticated `/api/trial-balance`), plus correlation fields:

```json
{
  "found": true,
  "format_version": 1,
  "source": "analee",
  "company_id": 123,
  "company_name": "Mokoena Traders (Pty) Ltd",
  "registration_number": "",
  "as_at": "2026-02-28",
  "period_start": "2025-03-01",
  "period_end": "2026-02-28",
  "rows": [
    { "link": "ca.810.001", "name": "Bank Cheque Account 1", "amount": 100.0 },
    { "link": "i.100.000", "name": "Sales", "amount": -100.0 }
  ],
  "balanced": true,
  "total_debits": 100.0,
  "total_credits": 100.0,
  "client_ref": "acc-7-42",
  "workspace_email": "client+acc-7-42@ws.theaccountants.local"
}
```

**Amount convention:** positive = debit, negative = credit; rows should sum to zero.

### Other responses

- `{"found": false}` — no active workspace user (**200**).
- **400** — not a workspace email, or no TB rows / missing company settings:

```json
{
  "found": true,
  "client_ref": "acc-7-42",
  "workspace_email": "client+…",
  "error": "No trial balance amounts for this period."
}
```

Map `rows` into existing Accountants TB intake (`dataimports/`, portal) using `link` as account key where supported.

---

## Suggested accountants implementation

1. **`analee_client.py`** (or `services/analee_provisioning.py`) — thin HTTP client:
   - `_headers()` → bearer + JSON
   - `ensure_workspace(client_ref, client_name, entity_name=None)`
   - `workspace_login_url(client_ref)` → returns `login_url` or composed URL
   - `fetch_workspace_trial_balance(client_ref)` → dict or raises on HTTP/contract errors
2. **Per-client model fields** (no Analee schema change): `analee_client_ref`, `analee_workspace_user_id`, `analee_workspace_email` (optional cache).
3. **Views:** “Open in Analee” → ensure + login-link + redirect; “Import trial balance from Analee” → trial-balance endpoint → existing import pipeline.
4. **Tests:** mock Analee HTTP; assert bearer header and path suffixes.

---

## Related Analee endpoints (out of scope for “three endpoints” but same secret)

| Endpoint | Purpose |
|----------|---------|
| `POST /api/provisioning/analee` | Grant/revoke **human** Analee access by email (`entitled` bool) — bundle / subscription sync. |

---

## Verification

On Analee after merge:

```bash
pytest tests/test_workspace_provisioning.py -q
python protected_assets.py --check
```

Manual (staging, provisioning enabled):

1. Ensure workspace for a test `client_ref`.
2. Login-link → open in browser → categorise bank lines.
3. Trial-balance POST → import rows in Accountants.

---

*Contract version: 2026-07-16 — matches `provisioning.py` workspace + TB routes.*
