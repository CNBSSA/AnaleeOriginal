"""Cash-basis chart guardrails."""
from services.cash_basis import (
    ACCRUAL_CONTROL_LINKS,
    REPAIRS_EXPENSE_LINKS,
    accrual_control_warning,
    capital_purchase_hint,
    is_accrual_control_link,
    is_ppe_cost_link,
    is_repairs_expense_link,
)


def test_accrual_control_links():
    assert 'ca.300.000' in ACCRUAL_CONTROL_LINKS
    assert 'cl.500.000' in ACCRUAL_CONTROL_LINKS
    assert is_accrual_control_link('ca.330.000')
    assert not is_accrual_control_link('i.100.000')


def test_ppe_cost_links():
    assert is_ppe_cost_link('na.040.000')
    assert not is_ppe_cost_link('na.040.001')
    assert not is_ppe_cost_link('e.451.002')


def test_repairs_expense_links():
    assert is_repairs_expense_link('e.451.002')
    assert not is_repairs_expense_link('na.040.000')


def test_accrual_warning_message():
    msg = accrual_control_warning('ca.300.000', 'Trade Receivables')
    assert msg is not None
    assert 'Cash-basis' in msg
    assert accrual_control_warning('i.100.000', 'Sales') is None


def test_capital_purchase_hint():
    ppe = capital_purchase_hint('na.040.000', 'Office Furniture - Cost')
    assert ppe is not None
    assert 'depreciation' in ppe.lower()
    repairs = capital_purchase_hint('e.451.002', 'Repairs - Furniture')
    assert repairs is not None
    assert 'cost' in repairs.lower()
