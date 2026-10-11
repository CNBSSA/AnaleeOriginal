"""Read-only analysis of bank statement lines for Financial Insights (Unit 2.3).

Uses description, amount, and date — not only ledger account categories — so
insights reflect what actually moved on the statement.
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from services.history_matching import normalize_description

_EMPTY_RESULT: Dict[str, Any] = {
    'empty': True,
    'message': 'No bank statement transactions in this period.',
    'top_spending': [],
    'unusual_amounts': [],
    'recurring_payments': [],
    'monthly_cash_flow': [],
    'cash_flow_trend': 'unknown',
    'totals': {'inflow': 0.0, 'outflow': 0.0, 'net': 0.0, 'count': 0},
}


def _parse_date(value) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if hasattr(value, 'year'):
        return datetime.combine(value, datetime.min.time())
    text = str(value).strip()
    if not text:
        return None
    for fmt in ('%Y-%m-%d', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S'):
        try:
            return datetime.strptime(text[:19], fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.replace('Z', '+00:00')[:19])
    except ValueError:
        return None


def _spending_label(row: Dict) -> str:
    amount = float(row.get('amount') or 0)
    if amount >= 0:
        return ''
    category = (row.get('category') or '').strip()
    account_name = (row.get('account_name') or '').strip()
    if category == 'Expenses' and account_name:
        return account_name
    if category == 'Expenses' and category:
        return category
    payee = normalize_description(row.get('description'))
    return payee or 'Uncategorized spending'


def analyze_statement_transactions(transactions: Sequence[Dict]) -> Dict[str, Any]:
    """Derive spending, anomalies, recurrence, and cash-flow trends from statement lines."""
    if not transactions:
        return dict(_EMPTY_RESULT)

    rows = [t for t in transactions if isinstance(t, dict)]
    if not rows:
        return dict(_EMPTY_RESULT)

    inflow = 0.0
    outflow = 0.0
    spending: Dict[str, Dict[str, float]] = defaultdict(lambda: {'total': 0.0, 'count': 0})
    payee_outflows: Dict[str, List[float]] = defaultdict(list)
    monthly: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {'inflow': 0.0, 'outflow': 0.0})

    all_outflow_amounts: List[float] = []

    for row in rows:
        amount = float(row.get('amount') or 0)
        if amount > 0:
            inflow += amount
        elif amount < 0:
            outflow += abs(amount)
            all_outflow_amounts.append(abs(amount))
            label = _spending_label(row)
            if label:
                spending[label]['total'] += abs(amount)
                spending[label]['count'] += 1
            payee = normalize_description(row.get('description'))
            if payee:
                payee_outflows[payee].append(abs(amount))

        when = _parse_date(row.get('date'))
        if when is not None:
            key = when.strftime('%Y-%m')
            if amount > 0:
                monthly[key]['inflow'] += amount
            elif amount < 0:
                monthly[key]['outflow'] += abs(amount)

    top_spending = sorted(
        [
            {'label': label, 'total': data['total'], 'count': int(data['count'])}
            for label, data in spending.items()
        ],
        key=lambda item: item['total'],
        reverse=True,
    )[:8]

    unusual_amounts: List[Dict[str, Any]] = []
    global_median = statistics.median(all_outflow_amounts) if all_outflow_amounts else 0.0

    for row in rows:
        amount = float(row.get('amount') or 0)
        if amount >= 0:
            continue
        abs_amt = abs(amount)
        payee = normalize_description(row.get('description')) or 'line'
        peer = payee_outflows.get(normalize_description(row.get('description')) or '', [])
        peer_median = statistics.median(peer) if len(peer) > 1 else global_median
        threshold = max(peer_median * 2.5, global_median * 3, 500.0)
        if abs_amt >= threshold and abs_amt > 0:
            unusual_amounts.append({
                'description': (row.get('description') or '')[:120],
                'amount': abs_amt,
                'reason': (
                    f'R{abs_amt:,.2f} is well above typical '
                    f'R{peer_median:,.2f} for this payee'
                    if len(peer) > 1
                    else f'R{abs_amt:,.2f} is a large outflow for the period'
                ),
            })
    unusual_amounts.sort(key=lambda item: item['amount'], reverse=True)
    unusual_amounts = unusual_amounts[:6]

    recurring_payments: List[Dict[str, Any]] = []
    for payee, amounts in payee_outflows.items():
        if len(amounts) < 2:
            continue
        typical = statistics.median(amounts)
        if typical <= 0:
            continue
        within = sum(1 for a in amounts if abs(a - typical) <= max(typical * 0.1, 5.0))
        if within >= 2:
            recurring_payments.append({
                'payee': payee,
                'count': len(amounts),
                'typical_amount': round(typical, 2),
            })
    recurring_payments.sort(key=lambda item: (item['count'], item['typical_amount']), reverse=True)
    recurring_payments = recurring_payments[:8]

    monthly_cash_flow = []
    for month in sorted(monthly.keys()):
        bucket = monthly[month]
        net = bucket['inflow'] - bucket['outflow']
        monthly_cash_flow.append({
            'month': month,
            'inflow': round(bucket['inflow'], 2),
            'outflow': round(bucket['outflow'], 2),
            'net': round(net, 2),
        })

    cash_flow_trend = 'stable'
    if len(monthly_cash_flow) >= 2:
        last_net = monthly_cash_flow[-1]['net']
        prev_net = monthly_cash_flow[-2]['net']
        if last_net > prev_net + 100:
            cash_flow_trend = 'improving'
        elif last_net < prev_net - 100:
            cash_flow_trend = 'declining'

    return {
        'empty': False,
        'message': '',
        'top_spending': top_spending,
        'unusual_amounts': unusual_amounts,
        'recurring_payments': recurring_payments,
        'monthly_cash_flow': monthly_cash_flow,
        'cash_flow_trend': cash_flow_trend,
        'totals': {
            'inflow': round(inflow, 2),
            'outflow': round(outflow, 2),
            'net': round(inflow - outflow, 2),
            'count': len(rows),
        },
    }


def format_statement_analysis_for_prompt(analysis: Dict[str, Any]) -> str:
    """Human-readable block fed to the AI alongside raw totals."""
    if analysis.get('empty'):
        return analysis.get('message') or _EMPTY_RESULT['message']

    lines = [
        'Bank statement pattern analysis (from actual lines in this period):',
        f"- Transactions: {analysis['totals']['count']}",
        f"- Money in: R{analysis['totals']['inflow']:,.2f}",
        f"- Money out: R{analysis['totals']['outflow']:,.2f}",
        f"- Net: R{analysis['totals']['net']:,.2f}",
        f"- Cash-flow trend (month-on-month): {analysis['cash_flow_trend']}",
    ]
    if analysis['top_spending']:
        lines.append('- Top spending (from statement activity):')
        for item in analysis['top_spending'][:5]:
            lines.append(
                f"  • {item['label']}: R{item['total']:,.2f} ({item['count']} lines)"
            )
    if analysis['recurring_payments']:
        lines.append('- Recurring outflows detected:')
        for item in analysis['recurring_payments'][:5]:
            lines.append(
                f"  • {item['payee']}: ~R{item['typical_amount']:,.2f} "
                f"({item['count']} times)"
            )
    if analysis['unusual_amounts']:
        lines.append('- Unusually large outflows:')
        for item in analysis['unusual_amounts'][:4]:
            lines.append(f"  • {item['description'][:60]}: {item['reason']}")
    if analysis['monthly_cash_flow']:
        lines.append('- Monthly net cash flow:')
        for bucket in analysis['monthly_cash_flow'][-6:]:
            lines.append(
                f"  • {bucket['month']}: R{bucket['net']:,.2f} "
                f"(in R{bucket['inflow']:,.2f}, out R{bucket['outflow']:,.2f})"
            )
    return '\n'.join(lines)


def cash_flow_analysis_from_statement(
    transactions: Sequence[Dict],
    analysis: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Structured cash-flow section for the Financial Insights page."""
    analysis = analysis or analyze_statement_transactions(transactions)
    if analysis.get('empty'):
        return {
            'current_status': analysis.get('message', _EMPTY_RESULT['message']),
            'projected_trend': 'No data for trend analysis',
            'key_drivers': [],
            'improvement_suggestions': [],
        }

    totals = analysis['totals']
    trend = analysis['cash_flow_trend']
    trend_text = {
        'improving': 'Recent months show improving net cash flow.',
        'declining': 'Recent months show declining net cash flow.',
        'stable': 'Net cash flow has been relatively stable month to month.',
        'unknown': 'Trend could not be determined.',
    }.get(trend, 'Trend could not be determined.')

    drivers = [
        f"Total money in: R{totals['inflow']:,.2f}",
        f"Total money out: R{totals['outflow']:,.2f}",
        f"Net for period: R{totals['net']:,.2f}",
    ]
    if analysis['top_spending']:
        top = analysis['top_spending'][0]
        drivers.append(
            f"Largest spending area on statements: {top['label']} "
            f"(R{top['total']:,.2f})"
        )

    suggestions: List[str] = []
    if analysis['unusual_amounts']:
        suggestions.append(
            'Review the unusually large outflows listed above before month-end.'
        )
    if analysis['recurring_payments']:
        suggestions.append(
            'Confirm recurring debit orders still match contracts and budgets.'
        )
    if not suggestions:
        suggestions.append(
            'Continue monitoring monthly net cash flow as new statements arrive.'
        )

    return {
        'current_status': f"Net cash flow: R{totals['net']:,.2f}",
        'projected_trend': trend_text,
        'key_drivers': drivers,
        'improvement_suggestions': suggestions,
    }
