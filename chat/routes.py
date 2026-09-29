"""
Financial AI Assistant Chat Routes
This module handles all chat-related functionality including message processing,
context management, and financial insights generation.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List
from flask import Blueprint, jsonify, request, render_template
from flask_login import login_required, current_user
from sqlalchemy import desc, or_

from models import db, Transaction, Account
from ai_insights import FinancialInsightsGenerator
from nlp_utils import get_claude_client as get_openai_client
from config import CLAUDE_MODEL

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the blueprint from __init__.py
from . import chat

@chat.route('/interface')
@login_required
def chat_interface():
    """Render the chat interface."""
    try:
        # Get unanalyzed transactions count
        unanalyzed_count = Transaction.query.filter(
            Transaction.user_id == current_user.id,
            or_(
                Transaction.account_id.is_(None),
                Transaction.explanation.is_(None)
            )
        ).count()

        logger.info(f"Found {unanalyzed_count} unanalyzed transactions for user {current_user.id}")
        return render_template('chat/chat_interface.html', unanalyzed_count=unanalyzed_count)
    except Exception as e:
        logger.error(f"Error loading chat interface: {str(e)}")
        return render_template('chat/chat_interface.html', error="Error loading transaction data")

@chat.route('/send', methods=['POST'])
@login_required
def send_message():
    """Process incoming chat messages and generate responses with financial context."""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()

        if not message:
            return jsonify({
                'success': False,
                'error': 'Empty message'
            })

        # Get OpenAI client
        client = get_openai_client()
        if not client:
            logger.error("Failed to initialize OpenAI client")
            return jsonify({
                'success': False,
                'error': 'AI service temporarily unavailable'
            })

        # Get user's financial context
        context = get_financial_context(current_user.id)
        logger.info(f"Processing chat message for user {current_user.id}")

        # Generate AI response with context
        try:
            response = generate_ai_response(client, message, context)
        except Exception as e:
            logger.error(f"Error generating AI response: {str(e)}")
            response = "I apologize, but I'm having trouble analyzing your request right now. Please try again in a moment."

        return jsonify({
            'success': True,
            'response': response,
            'context_update': context
        })

    except Exception as e:
        logger.error(f"Error processing chat message: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@chat.route('/history', methods=['GET'])
@login_required
def get_chat_history():
    """Retrieve chat history for the current user."""
    try:
        # For now, return empty history since we haven't implemented chat history storage yet
        # This prevents the 404 error while maintaining a valid response
        logger.info(f"Retrieving chat history for user {current_user.id}")
        return jsonify({
            'success': True,
            'history': []  # Will be implemented with chat history storage
        })
    except Exception as e:
        logger.error(f"Error retrieving chat history: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@chat.route('/context', methods=['GET'])
@login_required
def get_context():
    """Get current financial context for the chat."""
    try:
        context = get_financial_context(current_user.id)
        logger.info(f"Retrieved financial context for user {current_user.id}")
        return jsonify({
            'success': True,
            'context': context
        })
    except Exception as e:
        logger.error(f"Error retrieving financial context: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

# The assistant sees the client's whole FINANCIAL YEAR, not five lines.
# Before 2026-09-28 the context was the current calendar month's totals plus
# the five latest transactions — so a question about last quarter's rent was
# answered from five lines that did not contain it (QA, 2026-09-28).
CONTEXT_LINE_CAP = 150


def _period_for(user_id: int):
    """The client's current financial year (CompanySettings), else the last
    twelve months."""
    from models import CompanySettings
    settings = CompanySettings.query.filter_by(user_id=user_id).first()
    if settings is not None:
        try:
            fy = settings.get_financial_year()
            return fy['start_date'], fy['end_date'], 'financial year'
        except Exception:  # a settings row with no usable year-end
            pass
    end = datetime.now()
    return end - timedelta(days=365), end, 'last 12 months'


def get_financial_context(user_id: int) -> Dict:
    """
    The assistant's view of the books: the financial year's totals, a total
    per account, and the period's transactions (newest first, capped at
    CONTEXT_LINE_CAP with the cap stated), all in rand.
    """
    try:
        start_date, end_date, period_label = _period_for(user_id)

        transactions = Transaction.query.filter(
            Transaction.user_id == user_id,
            Transaction.date.between(start_date, end_date)
        ).order_by(desc(Transaction.date), desc(Transaction.id)).all()

        income = sum(t.amount for t in transactions if t.amount > 0)
        expenses = abs(sum(t.amount for t in transactions if t.amount < 0))
        balance = income - expenses

        by_account: Dict[str, Dict] = {}
        uncategorised = 0
        for t in transactions:
            name = t.account.name if t.account else 'Not yet categorised'
            if not t.account:
                uncategorised += 1
            row = by_account.setdefault(name, {'count': 0, 'total': 0.0})
            row['count'] += 1
            row['total'] += float(t.amount)
        account_totals = sorted(
            ({'account': k, 'count': v['count'], 'total': round(v['total'], 2)}
             for k, v in by_account.items()),
            key=lambda r: abs(r['total']), reverse=True)

        recent_transactions = [{
            'date': tx.date.strftime('%Y-%m-%d'),
            'description': tx.description,
            'amount': float(tx.amount),
            'category': tx.account.name if tx.account else 'Not yet categorised',
            'analyzed': bool(tx.account_id and tx.explanation)
        } for tx in transactions[:CONTEXT_LINE_CAP]]

        context = {
            'period_label': period_label,
            'period_start': start_date.strftime('%Y-%m-%d'),
            'period_end': end_date.strftime('%Y-%m-%d'),
            'income': float(income),
            'expenses': float(expenses),
            'balance': float(balance),
            'account_totals': account_totals,
            'uncategorised': uncategorised,
            'recent_transactions': recent_transactions,
            'total_transactions': len(transactions),
            'lines_shown': len(recent_transactions),
        }

        logger.info(f"Financial context generated for user {user_id}")
        return context

    except Exception as e:
        logger.error(f"Error getting financial context: {str(e)}")
        # Return a safe fallback context
        return {
            'period_label': '', 'period_start': '', 'period_end': '',
            'income': 0,
            'expenses': 0,
            'balance': 0,
            'account_totals': [],
            'uncategorised': 0,
            'recent_transactions': [],
            'total_transactions': 0,
            'lines_shown': 0,
            'error': str(e)
        }

def generate_ai_response(client, message: str, context: Dict) -> str:
    """Generate AI response with financial context."""
    try:
        # Create prompt with context
        prompt = f"""As a financial AI assistant, help the user with their query. Here's the current context:

Period: the {context.get('period_label') or 'period'} {context.get('period_start', '')} to {context.get('period_end', '')}.
All amounts are South African rand (R), from the company's own categorised bank statements.

Summary for the period:
- Money in: R{context['income']:,.2f}
- Money out: R{context['expenses']:,.2f}
- Net: R{context['balance']:,.2f}
- Transactions in the period: {context['total_transactions']} ({context.get('uncategorised', 0)} not yet categorised)

Totals by account (money in positive, money out negative):
{format_account_totals_for_prompt(context.get('account_totals', []))}

Transactions (newest first{', the latest ' + str(context['lines_shown']) + ' of ' + str(context['total_transactions']) if context.get('lines_shown', 0) < context['total_transactions'] else ''}):
{format_transactions_for_prompt(context['recent_transactions'])}

User Query: {message}

Answer from the figures above. Do not call the data dummy or inconsistent; if something is not in the period shown, say so."""

        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=512,
            system="You are a helpful financial assistant focused on providing clear, actionable advice based on the user's financial data.",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text.strip()

    except Exception as e:
        logger.error(f"Error generating AI response: {str(e)}")
        return "I apologize, but I'm having trouble generating a response right now. Please try again in a moment."

def format_transactions_for_prompt(transactions: List[Dict]) -> str:
    """Format transactions for the AI prompt (rand, with the account)."""
    return "\n".join([
        f"- {tx['date']}: {tx['description']} (R{tx['amount']:,.2f}) — {tx.get('category', '')}"
        for tx in transactions
    ]) or "- none in the period"


def format_account_totals_for_prompt(rows: List[Dict]) -> str:
    return "\n".join([
        f"- {r['account']}: R{r['total']:,.2f} ({r['count']} transactions)"
        for r in rows
    ]) or "- nothing categorised yet"