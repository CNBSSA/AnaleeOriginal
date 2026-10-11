"""Category / subcategory choices for chart-of-accounts forms (read-only vocabulary)."""

CHART_CATEGORIES = (
    'Assets',
    'Liabilities',
    'Equity',
    'Income',
    'Expenses',
)

# Matches the sub_category names used in services/chart_seed_data.py (read only).
CHART_SUBCATEGORIES = (
    'Current Asset',
    'Non-Current Asset',
    'Current Liability',
    'Non-Current Liabilities',
    'Equity',
    'Sales',
    'Income',
    'Cost of Sales',
    'Expenses',
    'Tax',
)
