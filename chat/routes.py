"""
Financial AI Assistant Chat Routes
This module handles all chat-related functionality including message processing,
context management, and financial insights generation.
"""

import logging
from typing import Dict, List, Optional
from flask import Blueprint, jsonify, request, render_template
from flask_login import login_required, current_user
from sqlalchemy import or_

from models import Transaction
from services.chat_statement_context import build_chat_context
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

        file_id = data.get('file_id')
        if file_id is not None:
            try:
                file_id = int(file_id)
            except (TypeError, ValueError):
                file_id = None

        context = get_financial_context(current_user.id, file_id=file_id)
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
        file_id = request.args.get('file_id', type=int)
        context = get_financial_context(current_user.id, file_id=file_id)
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

def get_financial_context(user_id: int, file_id: Optional[int] = None) -> Dict:
    """Read-only books + current bank statement for the assistant."""
    try:
        context = build_chat_context(user_id, file_id=file_id)
        logger.info(f"Financial context generated for user {user_id}")
        return context
    except Exception as e:
        logger.error(f"Error getting financial context: {str(e)}")
        return {
            'period_label': '', 'period_start': '', 'period_end': '',
            'income': 0,
            'expenses': 0,
            'balance': 0,
            'account_totals': [],
            'account_balances': [],
            'balance_accounts_shown': 0,
            'uncategorised': 0,
            'recent_transactions': [],
            'total_transactions': 0,
            'lines_shown': 0,
            'statement': None,
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

Totals by account in the period (money in positive, money out negative):
{format_account_totals_for_prompt(context.get('account_totals', []))}

Trial balance account balances for the same financial year (debit positive, credit negative):
{format_account_balances_for_prompt(context.get('account_balances', []))}

{format_statement_block_for_prompt(context.get('statement'))}

Financial-year transactions (newest first{', the latest ' + str(context['lines_shown']) + ' of ' + str(context['total_transactions']) if context.get('lines_shown', 0) < context['total_transactions'] else ''}):
{format_transactions_for_prompt(context['recent_transactions'])}

User Query: {message}

Answer from the figures above — especially the current bank statement when the question is about imported lines. Do not call the data dummy or inconsistent; if something is not in the data shown, say so. You cannot change accounts or explanations; only describe what is there."""

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


def format_account_balances_for_prompt(rows: List[Dict]) -> str:
    return "\n".join([
        f"- {r['account']} ({r.get('link', '')}): R{r['balance']:,.2f}"
        for r in rows
    ]) or "- no trial balance figures for this workspace yet"


def format_statement_block_for_prompt(statement: Optional[Dict]) -> str:
    if not statement:
        return (
            "Current bank statement: none uploaded yet — only the financial-year "
            "summary above is available."
        )
    cat = statement.get('categorization') or {}
    prov = statement.get('provenance') or {}
    cap_note = ''
    if statement.get('lines_shown', 0) < statement.get('transaction_count', 0):
        cap_note = (
            f", showing the latest {statement['lines_shown']} of "
            f"{statement['transaction_count']}"
        )
    lines = [
        "Current bank statement (read-only):",
        f"- File: {statement.get('filename', '')} "
        f"(uploaded {statement.get('upload_date', 'unknown')})",
        f"- Lines on this statement: {statement.get('transaction_count', 0)}",
        f"- Still needing account and/or explanation: "
        f"{statement.get('unprocessed_count', 0)}",
        "- Categorisation on this statement: "
        f"{cat.get('complete', 0)} complete, "
        f"{cat.get('needs_account', 0)} need an account, "
        f"{cat.get('needs_explanation', 0)} need an explanation, "
        f"{cat.get('needs_account_and_explanation', 0)} need both",
        "- Explanations by author on this statement: "
        f"accountant {prov.get('accountant', 0)}, "
        f"client {prov.get('client', 0)}, "
        f"Analee {prov.get('ai', 0)}, "
        f"other/unattributed {prov.get('unattributed', 0)}",
        f"Statement lines (newest first{cap_note}):",
    ]
    tx_lines = statement.get('transactions') or []
    if not tx_lines:
        lines.append("- no lines on this statement")
    else:
        for tx in tx_lines:
            lines.append(
                f"- {tx['date']}: {tx['description']} (R{tx['amount']:,.2f}) — "
                f"{tx.get('account', '')} [{tx.get('status', '')}]"
            )
    return "\n".join(lines)