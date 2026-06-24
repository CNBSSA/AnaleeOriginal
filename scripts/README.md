# Operational scripts

## `synthetic_monitor.py` — live uptime monitor

Logs in as a dedicated monitoring user and checks the **dashboard** and one
**report**, measuring latency. Exits non-zero (so a scheduler/CI run fails and
alerts) on any non-200, a bounce back to the login page, or a slow response.

### Run locally
```bash
pip install requests
MONITOR_BASE_URL=https://your-app.up.railway.app \
MONITOR_EMAIL=monitor@yourco.com \
MONITOR_PASSWORD='...' \
python scripts/synthetic_monitor.py
```

### Configuration (env vars)
| Variable | Required | Default | Purpose |
|---|---|---|---|
| `MONITOR_BASE_URL` | yes | — | Base URL of the deployed app |
| `MONITOR_EMAIL` | yes | — | Monitoring user's login email |
| `MONITOR_PASSWORD` | yes | — | Monitoring user's password |
| `MONITOR_REPORT_PATH` | no | `/trial-balance` | Report path to probe |
| `MONITOR_TIMEOUT` | no | `15` | Per-request timeout (seconds) |
| `MONITOR_MAX_SECONDS` | no | `5` | Slow-response threshold (seconds) |
| `MONITOR_ALERT_WEBHOOK` | no | — | Slack-style webhook for failure alerts |

### Scheduling
`.github/workflows/synthetic-monitor.yml` runs it every ~15 minutes once merged
to the default branch. Add the variables above as **repository secrets** first.
Create a dedicated, low-privilege monitoring user in the app (with company
settings configured so the report renders) — don't reuse a real customer login.
