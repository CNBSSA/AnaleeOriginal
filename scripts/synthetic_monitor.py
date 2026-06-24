#!/usr/bin/env python3
"""
Synthetic uptime monitor for the live Analee app.

Logs in as a dedicated monitoring user (handling the CSRF token), then fetches the
dashboard and one report, measuring response time. Exits non-zero (so a cron /
GitHub Action / Railway job fails and alerts) if any request is non-200, bounces
back to the login page, or exceeds the latency budget. Optionally posts to a
Slack-compatible webhook.

Configuration (environment variables):
  MONITOR_BASE_URL      required  e.g. https://your-app.up.railway.app
  MONITOR_EMAIL         required  login email of the monitoring user
  MONITOR_PASSWORD      required  its password
  MONITOR_REPORT_PATH   optional  report to probe (default /trial-balance)
  MONITOR_TIMEOUT       optional  per-request timeout seconds (default 15)
  MONITOR_MAX_SECONDS   optional  slow threshold seconds (default 5)
  MONITOR_ALERT_WEBHOOK optional  Slack-style webhook for failure alerts

Run:  python scripts/synthetic_monitor.py   (needs: pip install requests)
"""
import os
import re
import sys
import time

try:
    import requests
except ImportError:  # pragma: no cover
    print("synthetic_monitor: the 'requests' package is required", file=sys.stderr)
    sys.exit(2)

_CSRF_INPUT = re.compile(r'<input[^>]*name="csrf_token"[^>]*>', re.IGNORECASE)
_VALUE = re.compile(r'value="([^"]*)"', re.IGNORECASE)


def _env(name, default=None, required=False):
    value = os.environ.get(name, default)
    if required and not value:
        print(f"synthetic_monitor: missing required env {name}", file=sys.stderr)
        sys.exit(2)
    return value


def _extract_csrf(html):
    tag = _CSRF_INPUT.search(html or "")
    if not tag:
        return None
    match = _VALUE.search(tag.group(0))
    return match.group(1) if match else None


def _alert(webhook, message):
    if not webhook:
        return
    try:
        requests.post(webhook, json={"text": message}, timeout=10)
    except Exception as exc:  # pragma: no cover - best effort
        print(f"synthetic_monitor: alert webhook failed: {exc}", file=sys.stderr)


def check():
    base = _env("MONITOR_BASE_URL", required=True).rstrip("/")
    email = _env("MONITOR_EMAIL", required=True)
    password = _env("MONITOR_PASSWORD", required=True)
    report_path = _env("MONITOR_REPORT_PATH", "/trial-balance")
    timeout = float(_env("MONITOR_TIMEOUT", "15"))
    max_seconds = float(_env("MONITOR_MAX_SECONDS", "5"))
    webhook = _env("MONITOR_ALERT_WEBHOOK")

    failures = []
    session = requests.Session()
    session.headers["User-Agent"] = "analee-synthetic-monitor/1"

    # 1. Fetch the login page and its CSRF token.
    try:
        resp = session.get(base + "/auth/login", timeout=timeout)
    except Exception as exc:
        failures.append(f"GET /auth/login failed: {exc}")
        return _finish(failures, webhook)
    if resp.status_code != 200:
        failures.append(f"/auth/login -> {resp.status_code}")
    csrf = _extract_csrf(resp.text)

    # 2. Log in.
    data = {"email": email, "password": password}
    if csrf:
        data["csrf_token"] = csrf
    try:
        resp = session.post(base + "/auth/login", data=data,
                            timeout=timeout, allow_redirects=True)
    except Exception as exc:
        failures.append(f"POST /auth/login failed: {exc}")
        return _finish(failures, webhook)
    if resp.url.rstrip("/").endswith("/auth/login") or "Invalid email or password" in resp.text:
        failures.append("login failed (still on login page / invalid credentials)")
        return _finish(failures, webhook)

    # 3. Probe authenticated pages: dashboard + one report.
    for path in ["/dashboard", report_path]:
        try:
            start = time.monotonic()
            resp = session.get(base + path, timeout=timeout, allow_redirects=True)
            elapsed = time.monotonic() - start
        except Exception as exc:
            failures.append(f"GET {path} failed: {exc}")
            continue
        if resp.url.rstrip("/").endswith("/auth/login"):
            failures.append(f"{path} bounced to login (not authenticated)")
        elif resp.status_code != 200:
            failures.append(f"{path} -> {resp.status_code}")
        elif elapsed > max_seconds:
            failures.append(f"{path} slow: {elapsed:.2f}s > {max_seconds:.2f}s")
        else:
            print(f"OK   {path} ({resp.status_code}, {elapsed:.2f}s)")

    return _finish(failures, webhook)


def _finish(failures, webhook):
    if failures:
        message = "Analee synthetic monitor FAILED:\n- " + "\n- ".join(failures)
        print(message, file=sys.stderr)
        _alert(webhook, message)
        return 1
    print("Analee synthetic monitor: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(check())
