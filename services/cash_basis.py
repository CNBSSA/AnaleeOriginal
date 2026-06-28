"""Cash-basis categorisation guardrails for bank-statement analysis."""
from __future__ import annotations

# Accrual control accounts — rarely appropriate for cash-basis bank categorisation.
ACCRUAL_CONTROL_LINKS: frozenset[str] = frozenset({
    'ca.300.000',  # Trade Receivables
    'ca.300.001',  # Sundry Debtors
    'ca.330.000',  # Accrued Income
    'cl.500.000',  # Trade Payables
    'cl.600.000',  # Accrued Liabilities
})

# Repairs expense links — capital purchases should use PPE cost accounts instead.
REPAIRS_EXPENSE_LINKS: frozenset[str] = frozenset({
    'e.451.001',
    'e.451.002',
    'e.451.003',
    'e.451.004',
    'e.451.005',
})

PPE_COST_LINK_PREFIX = 'na.'


def is_accrual_control_link(link: str) -> bool:
    return link in ACCRUAL_CONTROL_LINKS


def is_ppe_cost_link(link: str) -> bool:
    """PPE cost accounts (non-current assets), excluding accumulated depreciation."""
    if not link.startswith(PPE_COST_LINK_PREFIX):
        return False
    return 'Acc. Depr' not in link and not link.endswith('.001')


def is_repairs_expense_link(link: str) -> bool:
    return link in REPAIRS_EXPENSE_LINKS


def accrual_control_warning(link: str, account_name: str) -> str | None:
    if not is_accrual_control_link(link):
        return None
    return (
        f'Cash-basis: "{account_name}" is an accrual control account. '
        'Categorise bank lines to income or expense instead. '
        'Export your trial balance to BooksXperts or The Accountants for AR/AP.'
    )


def capital_purchase_hint(link: str, account_name: str) -> str | None:
    if is_ppe_cost_link(link):
        return (
            f'PPE cost account selected. BooksXperts or The Accountants will handle '
            f'depreciation for "{account_name}".'
        )
    if is_repairs_expense_link(link):
        return (
            f'If this is a capital purchase (not routine maintenance), use the '
            f'**cost** account for the asset class (e.g. Office Furniture - Cost), '
            f'not "{account_name}". BooksXperts will depreciate.'
        )
    return None
