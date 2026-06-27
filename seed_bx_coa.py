"""Seed AdminChartOfAccounts with the BooksXperts standard chart of accounts.

Run once after deployment:
    DATABASE_URL=<railway-url> python seed_bx_coa.py

Idempotent — INSERT ON CONFLICT DO NOTHING, so re-runs are safe and existing
custom accounts added via the admin UI are never overwritten.

Source: CNBSSA/booksxpert app/management/commands/seed_chart_of_accounts.py
Entity types: Private Company (Pty) Ltd, Close Corporation, Sole Proprietor,
              NPO, Partnership — per create_entities.py.

Equity section uses Private Company names (most common SA entity type).
Other entity types share the same account_link codes in BooksXperts, so only
one set can exist here; an admin can rename the equity/equity-adjacent accounts
for their entity type after import.
"""
import sys

# ---------------------------------------------------------------------------
# Subcategory → (category, sub_category) mapping
# Mirrors BooksXperts create_subcategories.py PKs 1..10
# Values must fit String(50) in admin_chart_of_accounts
# ---------------------------------------------------------------------------
SUBCAT = {
    1:  ('Assets',       'Current Assets'),
    2:  ('Liabilities',  'Current Liabilities'),
    3:  ('Income',       'Cost of Sales'),
    4:  ('Expenses',     'Operating Expenses'),
    5:  ('Income',       'Revenue'),
    6:  ('Assets',       'Non-Current Assets'),
    7:  ('Liabilities',  'Non-Current Liabilities'),
    8:  ('Equity',       'Equity'),
    9:  ('Income',       'Tax'),
    10: ('Income',       'Other Income'),
}

# Subcategory IDs (same as BooksXperts)
CA  = 1   # Current Asset
CL  = 2   # Current Liability
COS = 3   # Cost of Sales
EXP = 4   # Expenses
SAL = 5   # Sales/Revenue
NCA = 6   # Non-Current Asset
NCL = 7   # Non-Current Liability
EQ  = 8   # Equity
TAX = 9   # Tax
INC = 10  # Income

# ---------------------------------------------------------------------------
# Accounts: (account_number, account_name, account_link, subcategory_id,
#            main_account_group)
# The main_account_group is stored in description for reference.
# ---------------------------------------------------------------------------
COMMON_ACCOUNTS = [
    # ── Current Assets ──────────────────────────────────────────────────────
    (1010, 'Bank Cheque Account 1',              'ca.810.001', CA,  'Cash and Cash Equivalent'),
    (1011, 'Bank Cheque Account 2',              'ca.810.002', CA,  'Cash and Cash Equivalent'),
    (1012, 'Bank Cheque Account 3',              'ca.810.003', CA,  'Cash and Cash Equivalent'),
    (1013, 'Trust Account',                      'ca.810.004', CA,  'Cash and Cash Equivalent'),
    (1020, 'Cash on hand',                       'ca.820.000', CA,  'Cash and Cash Equivalent'),
    (1021, 'Petty Cash',                         'ca.820.001', CA,  'Cash and Cash Equivalent'),
    (1100, 'Trade Receivables',                  'ca.300.000', CA,  'Trade and Other Receivables'),
    (1110, 'Sundry Debtors',                     'ca.300.001', CA,  'Trade and Other Receivables'),
    (1120, 'Prepaid Expenses',                   'ca.310.000', CA,  'Trade and Other Receivables'),
    (1130, 'Deposits',                           'ca.320.000', CA,  'Trade and Other Receivables'),
    (1140, 'Accrued Income',                     'ca.330.000', CA,  'Trade and Other Receivables'),
    (1150, 'Staff Loans',                        'ca.340.000', CA,  'Trade and Other Receivables'),
    (1160, 'Short-term Investments',             'ca.350.000', CA,  'Investment'),
    (1170, 'VAT Input Account',                  'ca.710.000', CA,  ''),
    (1175, 'VAT Suspense Account',               'ca.711.000', CA,  ''),
    (1180, 'Suspense Account',                   'ca.900.000', CA,  ''),
    (1200, 'Inventory - Raw Materials',          'ca.130.000', CA,  'Inventory'),
    (1210, 'Inventory - Finished Goods',         'ca.131.000', CA,  'Inventory'),
    (1220, 'Inventory - WIP',                    'ca.132.000', CA,  'Inventory'),
    # ── Non-Current Assets: PPE ─────────────────────────────────────────────
    (1300, 'Land and Buildings - Cost',          'na.010.000', NCA, 'Property Plant and Equipment'),
    (1301, 'Land and Buildings - Acc. Depr.',    'na.010.001', NCA, 'Property Plant and Equipment'),
    (1310, 'Office Furniture - Cost',            'na.040.000', NCA, 'Property Plant and Equipment'),
    (1311, 'Office Furniture - Acc. Depr.',      'na.040.001', NCA, 'Property Plant and Equipment'),
    (1320, 'Office Equipment - Cost',            'na.060.000', NCA, 'Property Plant and Equipment'),
    (1321, 'Office Equipment - Acc. Depr.',      'na.060.001', NCA, 'Property Plant and Equipment'),
    (1330, 'Computer Equipment - Cost',          'na.070.000', NCA, 'Property Plant and Equipment'),
    (1331, 'Computer Equipment - Acc. Depr.',    'na.070.001', NCA, 'Property Plant and Equipment'),
    (1340, 'Computer Software - Cost',           'na.080.000', NCA, 'Property Plant and Equipment'),
    (1341, 'Computer Software - Acc. Depr.',     'na.080.001', NCA, 'Property Plant and Equipment'),
    (1350, 'Motor Vehicles - Cost',              'na.030.000', NCA, 'Property Plant and Equipment'),
    (1351, 'Motor Vehicles - Acc. Depr.',        'na.030.001', NCA, 'Property Plant and Equipment'),
    (1360, 'Machinery and Tools - Cost',         'na.020.000', NCA, 'Property Plant and Equipment'),
    (1361, 'Machinery and Tools - Acc. Depr.',   'na.020.001', NCA, 'Property Plant and Equipment'),
    (1370, 'Leasehold Improvements - Cost',      'na.050.000', NCA, 'Property Plant and Equipment'),
    (1371, 'Leasehold Improvements - Acc. Depr.','na.050.001', NCA, 'Property Plant and Equipment'),
    (1380, 'Other Loose Assets - Cost',          'na.090.000', NCA, 'Property Plant and Equipment'),
    (1381, 'Other Loose Assets - Acc. Depr.',    'na.090.001', NCA, 'Property Plant and Equipment'),
    # Intangibles
    (1400, 'Goodwill',                           'na.100.000', NCA, 'Goodwill'),
    (1401, 'Goodwill - Amortisation',            'na.100.001', NCA, 'Goodwill'),
    (1410, 'Patents and Trademarks',             'na.110.000', NCA, 'Goodwill'),
    # Investment property
    (1450, 'Investment Property - Cost',         'na.150.000', NCA, 'Property Plant and Equipment'),
    (1451, 'Investment Property - Acc. Depr.',   'na.150.001', NCA, 'Property Plant and Equipment'),
    # Long-term investments
    (1500, 'Investment - Listed Shares',         'na.600.001', NCA, 'Investment'),
    (1510, 'Investment - Unlisted Shares',       'na.600.002', NCA, 'Investment'),
    (1520, 'Investment - Fixed Deposits',        'na.600.003', NCA, 'Investment'),
    (1525, 'Investment - Subsidiaries',          'na.600.004', NCA, 'Investment'),
    # ── Current Liabilities ────────────────────────────────────────────────
    (2000, 'Trade Payables',                     'cl.500.000', CL,  'Trade and Other Payable'),
    (2010, 'Sundry Creditors',                   'cl.500.001', CL,  'Trade and Other Payable'),
    (2020, 'Accrued Liabilities',                'cl.600.000', CL,  'Trade and Other Payable'),
    (2030, 'Deferred Income',                    'cl.610.000', CL,  'Trade and Other Payable'),
    (2040, 'Provision for Leave Pay',            'cl.620.000', CL,  'Trade and Other Payable'),
    (2045, 'Provision for Bonuses',              'cl.631.000', CL,  'Trade and Other Payable'),
    (2046, 'Provision for Warranty',             'cl.630.000', CL,  'Trade and Other Payable'),
    (2047, 'Provision for Pension Fund',         'cl.632.000', CL,  'Trade and Other Payable'),
    (2050, 'Credit Card Account 1',              'cl.810.001', CL,  'Cash and Cash Equivalent - credit'),
    (2051, 'Credit Card Account 2',              'cl.810.002', CL,  'Cash and Cash Equivalent - credit'),
    (2060, 'Bank Overdraft',                     'cl.810.099', CL,  'Cash and Cash Equivalent - credit'),
    (2100, 'Unsecured Loan - Current Portion',   'cl.200.100', CL,  'Trade and Other Payable'),
    (2110, 'Mortgage Bond - Current Portion',    'cl.201.000', CL,  'Trade and Other Payable'),
    (2120, 'Finance Lease - Current Portion',    'cl.202.000', CL,  'Trade and Other Payable'),
    # Statutory
    (2200, 'VAT Control (Output)',               'cl.710.000', CL,  'Taxes Owed'),
    (2210, 'PAYE Payable',                       'cl.350.000', CL,  'Taxes Owed'),
    (2220, 'UIF Payable',                        'cl.360.000', CL,  'Taxes Owed'),
    (2230, 'SDL Payable',                        'cl.370.000', CL,  'Taxes Owed'),
    (2240, 'Income Tax Payable',                 'cl.300.000', CL,  'Taxes Owed'),
    (2250, 'Provisional Tax Payable',            'cl.301.000', CL,  'Taxes Owed'),
    # ── Non-Current Liabilities ────────────────────────────────────────────
    (2500, 'Mortgage Bond',                      'nl.100.000', NCL, ''),
    (2510, 'Long-term Loan - Secured',           'nl.200.000', NCL, ''),
    (2520, 'Long-term Loan - Unsecured',         'nl.210.000', NCL, ''),
    (2530, 'Finance Lease Liability',            'nl.220.000', NCL, ''),
    (2540, 'Deferred Tax Liability',             'nl.230.000', NCL, ''),
    # ── Sales / Revenue ────────────────────────────────────────────────────
    (4000, 'Sales',                              'i.100.000',  SAL, 'Revenue'),
    (4005, 'Professional Fees',                  'i.080.000',  SAL, 'Revenue'),
    (4010, 'Sales - Services',                   'i.110.000',  SAL, 'Revenue'),
    (4020, 'Sales - Products',                   'i.120.000',  SAL, 'Revenue'),
    (4030, 'Sales - Exports',                    'i.130.000',  SAL, 'Revenue'),
    (4040, 'Sales Returns',                      'i.140.000',  SAL, 'Revenue'),
    (4050, 'Discount Allowed',                   'i.150.000',  SAL, 'Revenue'),
    (4100, 'Interest Income',                    'i.045.000',  SAL, ''),
    (4110, 'Rental Income',                      'i.060.000',  SAL, ''),
    (4120, 'Commission Received',                'i.070.000',  SAL, ''),
    (4130, 'Profit on Disposal of Asset',        'is.700.000', SAL, ''),
    (4140, 'Foreign Exchange Gain',              'is.710.000', SAL, ''),
    (4150, 'Sundry Income',                      'i.090.000',  SAL, ''),
    # ── Cost of Sales ──────────────────────────────────────────────────────
    (5000, 'Opening Stock',                      'cos.000.000', COS, 'Cost of Sales'),
    (5010, 'Purchases',                          'cos.002.000', COS, 'Cost of Sales'),
    (5020, 'Closing Stock',                      'cos.005.000', COS, 'Cost of Sales'),
    (5030, 'Direct Labour',                      'cos.010.000', COS, 'Cost of Sales'),
    (5040, 'Carriage Inwards',                   'cos.020.000', COS, 'Cost of Sales'),
    (5050, 'Subcontractor Costs',                'cos.030.000', COS, 'Cost of Sales'),
    (5060, 'Import Duties',                      'cos.040.000', COS, 'Cost of Sales'),
    (5070, 'Discount Received',                  'cos.050.000', COS, 'Cost of Sales'),
    # ── Expenses ───────────────────────────────────────────────────────────
    (6000, 'Accounting Fees',                    'e.301.000',  EXP, 'Accounting and Audit fees'),
    (6010, "Auditor's Remuneration - Fees",      'e.307.000',  EXP, 'Accounting and Audit fees'),
    (6011, "Auditor's Remuneration - Consulting",'e.307.001',  EXP, 'Accounting and Audit fees'),
    (6012, "Auditor's Remuneration - Taxation",  'e.307.002',  EXP, 'Accounting and Audit fees'),
    (6013, "Auditor's Remuneration - Expenses",  'e.307.003',  EXP, 'Accounting and Audit fees'),
    (6020, 'Admin and Management Fees',          'e.303.000',  EXP, ''),
    (6030, 'Advertising and Marketing',          'e.302.000',  EXP, ''),
    (6040, 'Assessment Rates and Municipal',     'e.306.000',  EXP, ''),
    (6050, 'Bad Debts',                          'e.320.000',  EXP, ''),
    (6060, 'Bank Charges',                       'e.321.000',  EXP, ''),
    (6070, 'Cleaning',                           'e.330.000',  EXP, ''),
    (6080, 'Commission Paid',                    'e.331.000',  EXP, ''),
    (6090, 'Computer Expenses',                  'e.332.000',  EXP, ''),
    (6100, 'Consulting Fees',                    'e.333.000',  EXP, ''),
    (6110, 'Consumables',                        'e.334.000',  EXP, ''),
    (6120, 'Debt Collection',                    'e.340.000',  EXP, ''),
    (6130, 'Delivery Expenses',                  'e.341.000',  EXP, ''),
    (6140, 'Discount Allowed (Exp.)',            'e.343.000',  EXP, ''),
    (6150, 'Donations - Tax Deductible',         'e.344.000',  EXP, ''),
    (6160, 'Donations - Non-deductible',         'e.344.009',  EXP, ''),
    (6170, 'Electricity and Water',              'e.350.000',  EXP, ''),
    (6180, 'Entertainment',                      'e.351.000',  EXP, ''),
    (6190, 'Fines and Penalties',                'e.359.009',  EXP, ''),
    (6200, 'Flowers and Office Decor',           'e.360.000',  EXP, ''),
    (6210, 'Gifts',                              'e.371.000',  EXP, ''),
    (6220, 'General Expenses - Deductible',      'e.370.000',  EXP, ''),
    (6230, 'Hire - Equipment',                   'e.380.000',  EXP, ''),
    (6240, 'Insurance',                          'e.390.000',  EXP, ''),
    (6250, 'Foreign Exchange Loss',              'e.395.000',  EXP, ''),
    # Finance costs
    (6300, 'Interest Paid - Long Term Loans',    'e.391.000',  EXP, 'Finance Cost'),
    (6310, 'Interest Paid - Bank Overdraft',     'e.391.001',  EXP, 'Finance Cost'),
    (6320, 'Interest Paid - Finance Lease',      'e.391.002',  EXP, 'Finance Cost'),
    (6330, 'Interest Paid - SARS Penalties',     'e.391.004',  EXP, 'Finance Cost'),
    # Operating leases
    (6400, 'Lease Rental - Premises',            'e.400.000',  EXP, 'Operating Leases'),
    (6410, 'Lease Rental - Motor Vehicles',      'e.400.001',  EXP, 'Operating Leases'),
    (6420, 'Lease Rental - Equipment',           'e.400.002',  EXP, 'Operating Leases'),
    (6430, 'Lease Rental - Other',               'e.400.003',  EXP, 'Operating Leases'),
    (6500, 'Legal Expense',                      'e.401.000',  EXP, ''),
    (6510, 'License and Permit',                 'e.420.000',  EXP, ''),
    (6520, 'Magazines and Books',                'e.412.000',  EXP, ''),
    (6530, 'Medical Expense',                    'e.413.000',  EXP, ''),
    (6540, 'Motor Vehicle Expense',              'e.415.000',  EXP, ''),
    (6550, 'Packaging',                          'e.430.000',  EXP, ''),
    (6560, 'Pension Costs',                      'e.431.000',  EXP, ''),
    (6570, 'Petrol and Oil',                     'e.433.000',  EXP, ''),
    (6580, 'Placement Fees',                     'e.434.000',  EXP, ''),
    (6590, 'Postage and Courier',                'e.436.000',  EXP, ''),
    (6600, 'Printing and Stationery',            'e.437.000',  EXP, ''),
    (6610, 'Promotions',                         'e.438.000',  EXP, ''),
    (6620, 'Protective Clothing and Uniforms',   'e.439.000',  EXP, ''),
    # Repairs
    (6700, 'Repairs - Motor Vehicles',           'e.451.001',  EXP, 'Repairs and Equipment'),
    (6710, 'Repairs - Furniture',                'e.451.002',  EXP, 'Repairs and Equipment'),
    (6720, 'Repairs - Machinery and Tools',      'e.451.003',  EXP, 'Repairs and Equipment'),
    (6730, 'Repairs - Computer and IT',          'e.451.004',  EXP, 'Repairs and Equipment'),
    (6740, 'Repairs - Building and Premises',    'e.451.005',  EXP, 'Repairs and Equipment'),
    (6750, 'Research and Development',           'e.453.000',  EXP, ''),
    # Salaries / staff costs
    (6800, 'Salaries',                           'e.460.000',  EXP, 'Salary Costs'),
    (6810, 'Wages - Full-time',                  'e.510.000',  EXP, 'Salary Costs'),
    (6820, 'Wages - Casual',                     'e.510.001',  EXP, 'Salary Costs'),
    (6830, "UIF - Employer's Contribution",      'e.510.002',  EXP, 'Salary Costs'),
    (6840, 'SDL - Skills Development Levy',      'e.510.003',  EXP, 'Salary Costs'),
    (6845, "Workmen's Compensation (COIDA)",     'e.510.004',  EXP, 'Salary Costs'),
    (6850, 'Employee Benefits',                  'e.470.000',  EXP, 'Salary Costs'),
    (6855, 'Travel Allowance',                   'e.483.001',  EXP, 'Salary Costs'),
    (6860, 'Cell Phone / Communication Allowance','e.480.001', EXP, 'Salary Costs'),
    (6865, 'Staff Welfare',                      'e.465.000',  EXP, 'Salary Costs'),
    (6870, 'Training and Development',           'e.482.000',  EXP, 'Salary Costs'),
    (6900, 'Secretarial Fees',                   'e.461.000',  EXP, ''),
    (6910, 'Security',                           'e.462.000',  EXP, ''),
    (6920, 'Software Expenses',                  'e.464.000',  EXP, ''),
    (6930, 'Subscriptions',                      'e.466.000',  EXP, ''),
    (6940, 'Telephone and Internet',             'e.480.000',  EXP, ''),
    (6950, 'Transport',                          'e.481.000',  EXP, ''),
    (6960, 'Travel - Local',                     'e.483.000',  EXP, ''),
    (6970, 'Travel - Overseas',                  'e.484.000',  EXP, ''),
    (6980, 'Website and Hosting',                'e.420.011',  EXP, ''),
    # Depreciation expense lines
    (6990, 'Depreciation - Land and Buildings',  'e.810.000',  EXP, 'Depreciation Expenses'),
    (6991, 'Depreciation - Office Furniture',    'e.840.000',  EXP, 'Depreciation Expenses'),
    (6992, 'Depreciation - Office Equipment',    'e.860.000',  EXP, 'Depreciation Expenses'),
    (6993, 'Depreciation - Computer Equipment',  'e.870.000',  EXP, 'Depreciation Expenses'),
    (6994, 'Depreciation - Computer Software',   'e.880.000',  EXP, 'Depreciation Expenses'),
    (6995, 'Depreciation - Motor Vehicles',      'e.830.000',  EXP, 'Depreciation Expenses'),
    (6996, 'Depreciation - Machinery',           'e.820.000',  EXP, 'Depreciation Expenses'),
    (6997, 'Depreciation - Leasehold Improvements','e.850.000',EXP, 'Depreciation Expenses'),
    (6998, 'Depreciation - Investment Property', 'e.895.000',  EXP, 'Depreciation Expenses'),
    (6999, 'Depreciation - Other Assets',        'e.890.000',  EXP, 'Depreciation Expenses'),
    # ── Tax (below the line) ───────────────────────────────────────────────
    (8000, 'Provision for Income Tax',           't.900.000',  TAX, ''),
    (8010, 'Deferred Tax',                       't.910.000',  TAX, ''),
]

# Equity section — Private Company (Pty) Ltd names used for the shared link codes.
# Other entity types use the same account_link values in BooksXperts; rename
# these accounts via the admin UI if the entity type differs.
#
# Entity types recognised by BooksXperts:
#   Private Company (Pty) Ltd  — Share Capital, Retained Earnings, Dividends
#   Close Corporation (CC)     — Members' Contribution, Members' Interest
#   Sole Proprietor            — Capital Account, Drawings Account
#   Partnership                — Capital Contributions (per-partner accounts dynamic)
#   NPO                        — Accumulated Funds, Restricted/Unrestricted Funds
PRIVATE_COMPANY_EQUITY = [
    (7000, 'Share Capital - Ordinary',    'q.100.000', EQ,  'Entity: Private Company — rename for CC/SP/NPO/Partnership'),
    (7010, 'Share Capital - Preference',  'q.110.000', EQ,  'Entity: Private Company — rename for CC/NPO'),
    (7020, 'Share Premium',               'q.120.000', EQ,  'Entity: Private Company — rename for NPO (Unrestricted Funds)'),
    (7030, 'Capital Contribution',        'q.020.000', EQ,  'Entity: Private Company — also used by SP and Partnership'),
    (7040, 'Retained Earnings',           'q.200.000', EQ,  'Entity: Private Company — rename for CC/Partnership; NPO: Accumulated Funds'),
    (7050, 'Dividends Declared',          'q.210.000', EQ,  'Entity: Private Company — rename for CC (Distributions to Members)'),
    (7060, 'Current Year Profit/Loss',    'q.220.000', EQ,  'Entity: all types — Current Year Profit / (Loss)'),
    (2560, "Directors' Loan - Long-term", 'nl.300.001', NCL, 'Entity: Private Company — rename for CC (Member Loan) or SP (Owner Loan)'),
    (2570, "Directors' Loan - Short-term",'cl.300.001', CL,  'Entity: Private Company — rename for CC (Member Loan - Short-term)'),
]


def main():
    """Seed AdminChartOfAccounts from the BooksXperts standard chart."""
    from app import create_app
    from models import db, AdminChartOfAccounts

    app = create_app()

    with app.app_context():
        try:
            all_rows = COMMON_ACCOUNTS + PRIVATE_COMPANY_EQUITY
            inserted = 0
            skipped = 0

            for (num, name, link, sub_id, main_group) in all_rows:
                category, sub_category = SUBCAT[sub_id]
                description = main_group if main_group else None

                if AdminChartOfAccounts.query.filter_by(link=link).first():
                    skipped += 1
                    continue

                db.session.add(AdminChartOfAccounts(
                    link=link,
                    code=str(num),
                    name=name,
                    category=category,
                    sub_category=sub_category,
                    description=description,
                ))
                inserted += 1

            db.session.commit()
            print(f"Seed complete: {inserted} inserted, {skipped} already existed "
                  f"(total in template: {len(all_rows)}).")

        except Exception as exc:
            db.session.rollback()
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)


if __name__ == '__main__':
    main()
