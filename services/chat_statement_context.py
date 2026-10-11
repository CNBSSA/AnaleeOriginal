"""Read-only statement + books context for the AI Assistant Chat (Unit 2.2).

The chat explains and finds; it never writes categorisations or explanations.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional

from sqlalchemy import desc

from models import CompanySettings, Transaction, UploadedFile, transaction_in_financial_period
from reports.trial_balance_service import load_trial_balance
from services.analyze_processing import (
    count_unprocessed_transactions,
    has_real_account_assignment,
    provenance_summary,
    transaction_needs_processing,
)
from services.workspace_fy import current_financial_year_period

# Keep prompts bounded; the cap is stated in the assistant prompt.
STATEMENT_LINE_CAP = 150
BALANCE_LINE_CAP = 40


def _period_for(user_id: int):
    settings = CompanySettings.query.filter_by(user_id=user_id).first()
    if settings is not None:
        try:
            start_date, end_exclusive, end_inclusive, start_year = (
                current_financial_year_period(user_id, settings))
            return (
                start_date,
                end_exclusive,
                end_inclusive,
                start_year,
                'financial year',
                settings,
            )
        except Exception:
            pass
    end = datetime.now()
    return end - timedelta(days=365), end, end, None, 'last 12 months', settings


def resolve_statement_file(user_id: int, file_id: Optional[int] = None) -> Optional[UploadedFile]:
    if file_id is not None:
        found = UploadedFile.query.filter_by(id=file_id, user_id=user_id).first()
        if found is not None:
            return found
    return (
        UploadedFile.query.filter_by(user_id=user_id)
        .order_by(desc(UploadedFile.upload_date), desc(UploadedFile.id))
        .first()
    )


def _categorization_row(tx: Transaction) -> Dict:
    has_account = has_real_account_assignment(tx)
    has_explanation = bool(tx.explanation and tx.explanation.strip())
    if has_account and has_explanation:
        status = 'complete'
    elif not has_account and not has_explanation:
        status = 'needs account and explanation'
    elif not has_account:
        status = 'needs account'
    else:
        status = 'needs explanation'
    account_name = tx.account.name if tx.account else 'Not yet categorised'
    return {
        'date': tx.date.strftime('%Y-%m-%d') if tx.date else '',
        'description': tx.description or '',
        'amount': float(tx.amount),
        'account': account_name,
        'has_real_account': has_account,
        'has_explanation': has_explanation,
        'status': status,
    }


def _statement_categorization_counts(transactions: List[Transaction]) -> Dict[str, int]:
    complete = 0
    needs_account = 0
    needs_explanation = 0
    needs_both = 0
    for tx in transactions:
        has_account = has_real_account_assignment(tx)
        has_explanation = bool(tx.explanation and tx.explanation.strip())
        if has_account and has_explanation:
            complete += 1
        elif not has_account and not has_explanation:
            needs_both += 1
        elif not has_account:
            needs_account += 1
        else:
            needs_explanation += 1
    return {
        'complete': complete,
        'needs_account': needs_account,
        'needs_explanation': needs_explanation,
        'needs_account_and_explanation': needs_both,
        'still_to_process': sum(
            1 for tx in transactions if transaction_needs_processing(tx)),
    }


def _trial_balance_snapshot(user_id: int, end_inclusive: datetime, settings) -> List[Dict]:
    if settings is None:
        return []
    try:
        ctx = load_trial_balance(user_id, as_at=end_inclusive)
    except ValueError:
        return []
    rows = sorted(ctx.rows, key=lambda r: abs(r.amount), reverse=True)
    shown = rows[:BALANCE_LINE_CAP]
    return [
        {
            'link': row.link,
            'account': row.account_name,
            'balance': float(row.amount),
        }
        for row in shown
    ]


def build_chat_context(user_id: int, file_id: Optional[int] = None) -> Dict:
    """Workspace FY totals, trial-balance balances, and the current statement."""
    start_date, end_exclusive, end_inclusive, _fy_start, period_label, settings = _period_for(
        user_id)

    fy_transactions = Transaction.query.filter(
        Transaction.user_id == user_id,
        transaction_in_financial_period(Transaction.date, start_date, end_exclusive),
    ).order_by(desc(Transaction.date), desc(Transaction.id)).all()

    income = sum(t.amount for t in fy_transactions if t.amount > 0)
    expenses = abs(sum(t.amount for t in fy_transactions if t.amount < 0))
    balance = income - expenses

    by_account: Dict[str, Dict] = {}
    uncategorised = 0
    for t in fy_transactions:
        if not has_real_account_assignment(t):
            uncategorised += 1
        name = t.account.name if t.account else 'Not yet categorised'
        row = by_account.setdefault(name, {'count': 0, 'total': 0.0})
        row['count'] += 1
        row['total'] += float(t.amount)
    account_totals = sorted(
        ({'account': k, 'count': v['count'], 'total': round(v['total'], 2)}
         for k, v in by_account.items()),
        key=lambda r: abs(r['total']), reverse=True)

    recent_transactions = [
        {
            'date': tx.date.strftime('%Y-%m-%d'),
            'description': tx.description,
            'amount': float(tx.amount),
            'category': tx.account.name if tx.account else 'Not yet categorised',
            'analyzed': not transaction_needs_processing(tx),
        }
        for tx in fy_transactions[:STATEMENT_LINE_CAP]
    ]

    statement_file = resolve_statement_file(user_id, file_id)
    statement_block: Optional[Dict] = None
    if statement_file is not None:
        stmt_rows = (
            Transaction.query.filter_by(file_id=statement_file.id, user_id=user_id)
            .order_by(desc(Transaction.date), desc(Transaction.id))
            .all()
        )
        stmt_lines = [_categorization_row(tx) for tx in stmt_rows[:STATEMENT_LINE_CAP]]
        statement_block = {
            'file_id': statement_file.id,
            'filename': statement_file.filename,
            'upload_date': (
                statement_file.upload_date.strftime('%Y-%m-%d')
                if statement_file.upload_date else ''),
            'transaction_count': len(stmt_rows),
            'lines_shown': len(stmt_lines),
            'unprocessed_count': count_unprocessed_transactions(
                statement_file.id, user_id),
            'categorization': _statement_categorization_counts(stmt_rows),
            'provenance': provenance_summary(statement_file.id, user_id),
            'transactions': stmt_lines,
        }

    account_balances = _trial_balance_snapshot(user_id, end_inclusive, settings)

    return {
        'period_label': period_label,
        'period_start': start_date.strftime('%Y-%m-%d'),
        'period_end': end_inclusive.strftime('%Y-%m-%d'),
        'income': float(income),
        'expenses': float(expenses),
        'balance': float(balance),
        'account_totals': account_totals,
        'account_balances': account_balances,
        'balance_accounts_shown': len(account_balances),
        'uncategorised': uncategorised,
        'recent_transactions': recent_transactions,
        'total_transactions': len(fy_transactions),
        'lines_shown': len(recent_transactions),
        'statement': statement_block,
    }
