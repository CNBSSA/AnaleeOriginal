"""Default financial-year workspace when the current year has no transactions.

Dashboard, chat context, and Financial Insights used to always scope to
``get_financial_year()`` (today's financial year). When that year is empty but
older years have data, the UI showed zeros while the books were not empty.
"""
from __future__ import annotations

from datetime import date, datetime

from models import Transaction, db


def fy_start_year_for_moment(moment: date | datetime, fy_end: int) -> int:
    """Start year of the financial year containing ``moment`` (dashboard convention)."""
    if isinstance(moment, date) and not isinstance(moment, datetime):
        moment = datetime.combine(moment, datetime.min.time())
    if fy_end == 12:
        return moment.year
    return moment.year if moment.month > fy_end else moment.year - 1


def _fy_end_from_settings(company_settings) -> int:
    if company_settings is None:
        return 12
    return int(company_settings.financial_year_end or 12)


def fy_start_years_with_transactions(user_id: int, fy_end: int) -> set[int]:
    years: set[int] = set()
    for (tx_date,) in db.session.query(Transaction.date).filter(
        Transaction.user_id == user_id,
    ):
        if tx_date:
            years.add(fy_start_year_for_moment(tx_date, fy_end))
    return years


def default_fy_start_year(user_id: int, company_settings) -> int:
    """Prefer the current financial year when it has data; else the latest year that does."""
    fy_end = _fy_end_from_settings(company_settings)
    current = fy_start_year_for_moment(datetime.now(), fy_end)
    with_data = fy_start_years_with_transactions(user_id, fy_end)
    if not with_data:
        return current
    if current in with_data:
        return current
    return max(with_data)


def bounds_for_fy_start(company_settings, start_year: int) -> dict:
    from reports.routes import _financial_year_bounds

    return _financial_year_bounds(company_settings, start_year)


def current_financial_year_period(user_id: int, company_settings):
    """``(start_date, end_exclusive, end_inclusive)`` for the workspace default FY."""
    start_year = default_fy_start_year(user_id, company_settings)
    bounds = bounds_for_fy_start(company_settings, start_year)
    return bounds['start'], bounds['end_exclusive'], bounds['end_inclusive'], start_year
