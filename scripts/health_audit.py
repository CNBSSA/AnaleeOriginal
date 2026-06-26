#!/usr/bin/env python3
"""
Static health audit for Analee — a "morning health check" adapted from a
Django-flavoured spec to this Flask + SQLAlchemy codebase (tenancy == user_id).

Scans the source for the confidence-killers and prints a baseline report:

  Objective 1  IFRS    money columns stored as Float instead of Numeric/Decimal
  Objective 2  Tenancy unscoped owned-entity queries (no user_id / current_user)
  Objective 3  Deploy  hardcoded secrets; writes to the ephemeral container FS
  Objective 4  DB      unguarded .first()/.get() dereference; N+1 query shapes

This is a REPORT, not a CI gate (it exits 0). Use it for a baseline -> fix ->
re-measure pass. The dynamic cross-tenant proof lives in
tests/test_multitenant_isolation.py.

Usage:  python scripts/health_audit.py
"""
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {".git", "migrations", "ignore", "tests", "venv", ".venv", "__pycache__", "scripts"}
OWNED_MODELS = ("Account", "Transaction", "UploadedFile", "FinancialGoal",
                "AlertConfiguration", "AlertHistory", "CompanySettings",
                "RiskAssessment", "FinancialRecommendation", "HistoricalData",
                "BankStatementUpload")
MONEY_HINTS = ("amount", "balance", "threshold", "target", "current_amount",
               "price", "total", "value", "debit", "credit", "opening", "closing")


def _py_files():
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            if name.endswith(".py"):
                yield os.path.join(root, name)


def _rel(path):
    return os.path.relpath(path, REPO)


def _lines(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.readlines()


def audit_money_floats():
    """Objective 1: monetary columns declared as Float (should be Numeric/Decimal)."""
    findings = []
    col_re = re.compile(r"(\w+)\s*=\s*Column\(\s*Float", re.IGNORECASE)
    for path in _py_files():
        if os.path.basename(path) != "models.py":
            continue
        for i, line in enumerate(_lines(path), 1):
            m = col_re.search(line)
            if m and any(h in m.group(1).lower() for h in MONEY_HINTS):
                findings.append((_rel(path), i, m.group(1), line.strip()))
    return findings


def audit_hardcoded_secrets():
    """Objective 3: hardcoded credentials / keys (should come from env)."""
    findings = []
    patterns = [
        (re.compile(r"set_password\(\s*['\"][^'\"]+['\"]\s*\)"), "hardcoded password literal"),
        (re.compile(r"(SECRET_KEY|API_KEY|PASSWORD|TOKEN)\s*=\s*['\"][^'\"]{6,}['\"]", re.I), "hardcoded secret assignment"),
        (re.compile(r"sk-ant-[A-Za-z0-9\-_]{6,}"), "Anthropic key literal"),
        (re.compile(r"sk-[A-Za-z0-9]{20,}"), "OpenAI-style key literal"),
        (re.compile(r"postgres(ql)?://[^'\"\s]+:[^'\"\s]+@"), "DB URL with inline credentials"),
    ]
    for path in _py_files():
        for i, line in enumerate(_lines(path), 1):
            if "os.environ" in line or "os.getenv" in line:
                continue
            for rx, label in patterns:
                if rx.search(line):
                    findings.append((_rel(path), i, label, line.strip()[:120]))
    return findings


def audit_ephemeral_writes():
    """Objective 3: writes to the local container FS (lost on Railway restarts)."""
    findings = []
    patterns = [
        (re.compile(r"\.save\(\s*['\"]?/tmp/"), "save() to /tmp (transient)"),
        (re.compile(r"\bopen\([^)]*,\s*['\"][wa]b?['\"]"), "open(..., 'w'/'a') local write"),
        (re.compile(r"FileHandler\("), "logging.FileHandler (file lost on restart)"),
        (re.compile(r"os\.path\.join\(\s*['\"]/tmp"), "path under /tmp"),
    ]
    for path in _py_files():
        for i, line in enumerate(_lines(path), 1):
            for rx, label in patterns:
                if rx.search(line):
                    findings.append((_rel(path), i, label, line.strip()[:120]))
    return findings


def audit_unscoped_queries():
    """Objective 2: owned-entity lookups by id with no user_id / current_user nearby."""
    findings = []
    owned = "|".join(OWNED_MODELS)
    lookup_re = re.compile(rf"\b({owned})\.query\.(get|get_or_404)\(")
    for path in _py_files():
        lines = _lines(path)
        for i, line in enumerate(lines, 1):
            m = lookup_re.search(line)
            if not m:
                continue
            window = "".join(lines[max(0, i - 1): i + 6])  # this line + next ~5
            if "user_id" not in window and "current_user" not in window:
                findings.append((_rel(path), i, m.group(1), line.strip()[:120]))
    return findings


def audit_unguarded_deref():
    """Objective 4: ORM .first()/.get(id) result dereferenced without a None check.

    Only flags real None risks: `.first().attr` and single-argument `.get(x).attr`.
    Skips `.get('k', default)` (a default makes it safe) and the *_or_404 helpers.
    """
    findings = []
    first_rx = re.compile(r"\.first\(\)\s*[.\[]\s*\w")          # .first().attr / .first()[...]
    get_rx = re.compile(r"\.get\(\s*[^,()]+\)\s*[.\[]\s*\w")    # .get(single simple arg).attr (no default, no nested call)
    for path in _py_files():
        for i, line in enumerate(_lines(path), 1):
            if "get_or_404" in line or "first_or_404" in line:
                continue
            if first_rx.search(line) or get_rx.search(line):
                findings.append((_rel(path), i, "deref without None-check", line.strip()[:120]))
    return findings


def _print(title, findings, good="none found"):
    print(f"\n=== {title} ===")
    if not findings:
        print(f"  OK: {good}")
        return 0
    for f in findings:
        rel, line = f[0], f[1]
        detail = " | ".join(str(x) for x in f[2:])
        print(f"  {rel}:{line}  {detail}")
    print(f"  -> {len(findings)} finding(s)")
    return len(findings)


def main():
    print("Analee static health audit (baseline)")
    print("=" * 50)
    total = 0
    total += _print("Objective 1 — money columns as Float (IFRS: use Numeric/Decimal)",
                    audit_money_floats())
    total += _print("Objective 2 — owned-entity id lookups without a tenant (user_id) filter",
                    audit_unscoped_queries(),
                    good="every owned-entity id lookup is tenant-scoped")
    total += _print("Objective 3 — hardcoded secrets (move to env vars)",
                    audit_hardcoded_secrets())
    total += _print("Objective 3 — writes to ephemeral container FS (use external storage)",
                    audit_ephemeral_writes())
    total += _print("Objective 4 — .first()/.get() dereferenced without a None check",
                    audit_unguarded_deref())
    print("\n" + "=" * 50)
    print(f"Total findings: {total}")
    print("Note: this is a report; review each finding. Cross-tenant isolation is "
          "proven dynamically by tests/test_multitenant_isolation.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
