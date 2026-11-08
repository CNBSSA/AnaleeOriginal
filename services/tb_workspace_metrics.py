"""Caller-side trial-balance metrics for dashboard and reports (Work #11025).

Calls the locked ``load_trial_balance`` service; never reimplements TB math.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, Optional

from models import CompanySettings
from reports.routes import (
    ASSET_CATEGORY,
    EXPENSE_CATEGORY,
    INCOME_CATEGORY,
    _account_category,
)
from reports.trial_balance_service import BANK_LINK_PREFIXES, load_trial_balance
from services.workspace_fy import bounds_for_fy_start


def _link_is_bank(link: str) -> bool:
    prefix = (link or '').strip()
    return any(prefix.startswith(p) for p in BANK_LINK_PREFIXES)


def trial_balance_metrics_for_fy(
    user_id: int,
    company_settings: CompanySettings | None,
    start_year: int,
) -> Dict[str, Any]:
    """Income, expenses, net and bank balances for one financial year."""
    bounds = bounds_for_fy_start(company_settings, start_year)
    end_inclusive = bounds['end_inclusive']
    try:
        ctx = load_trial_balance(user_id, as_at=end_inclusive)
    except ValueError:
        return {
            'total_income': 0.0,
            'total_expenses': 0.0,
            'net': 0.0,
            'bank_balances_by_link': {},
            'ctx': None,
        }

    amounts_by_link = {row.link: row.amount for row in ctx.rows}
    link_to_category = {
        acct.link: _account_category(acct) for acct in ctx.accounts
    }

    total_income = Decimal('0')
    total_expenses = Decimal('0')
    bank_balances: Dict[str, float] = {}

    for link, amount in amounts_by_link.items():
        category = link_to_category.get(link, '')
        if category == INCOME_CATEGORY:
            total_income += -amount
        elif category == EXPENSE_CATEGORY:
            total_expenses += amount if amount > 0 else -amount
        if category == ASSET_CATEGORY and _link_is_bank(link):
            bank_balances[link] = float(amount)

    return {
        'total_income': float(total_income),
        'total_expenses': float(total_expenses),
        'net': float(total_income - total_expenses),
        'bank_balances_by_link': bank_balances,
        'ctx': ctx,
    }
