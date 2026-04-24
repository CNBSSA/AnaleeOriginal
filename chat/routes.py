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

def get_financial_context(user_id: int) -> Dict:
    """
    Get current financial context including recent transactions,
    monthly summary, and key metrics.
    """
    try:
        # Get current month's transactions
        start_date = datetime.now().replace(day=1)
        end_date = (start_date + timedelta(days=32)).replace(day=1) - timedelta(days=1)

        transactions = Transaction.query.filter(
            Transaction.user_id == user_id,
            Transaction.date.between(start_date, end_date)
        ).order_by(desc(Transaction.date)).all()

        # Calculate monthly totals
        income = sum(t.amount for t in transactions if t.amount > 0)
        expenses = abs(sum(t.amount for t in transactions if t.amount < 0))
        balance = income - expenses

        # Get recent transactions
        recent = Transaction.query.filter_by(user_id=user_id)\
            .order_by(desc(Transaction.date))\
            .limit(5)\
            .all()

        recent_transactions = [{
            'date': tx.date.strftime('%Y-%m-%d'),
            'description': tx.description,
            'amount': float(tx.amount),
            'category': tx.account.category if tx.account else 'Uncategorized',
            'analyzed': bool(tx.account_id and tx.explanation)
        } for tx in recent]

        context = {
            'income': float(income),
            'expenses': float(expenses),
            'balance': float(balance),
            'recent_transactions': recent_transactions,
            'total_transactions': len(transactions)
        }

        logger.info(f"Financial context generated for user {user_id}")
        return context

    except Exception as e:
        logger.error(f"Error getting financial context: {str(e)}")
        # Return a safe fallback context
        return {
            'income': 0,
            'expenses': 0,
            'balance': 0,
            'recent_transactions': [],
            'total_transactions': 0,
            'error': str(e)
        }

def generate_ai_response(client, message: str, context: Dict) -> str:
    """Generate AI response with financial context."""
    try:
        # Create prompt with context
        prompt = f"""As a financial AI assistant, help the user with their query. Here's the current context:

Monthly Summary:
- Income: ${context['income']:,.2f}
- Expenses: ${context['expenses']:,.2f}
- Balance: ${context['balance']:,.2f}

Recent Transactions:
{format_transactions_for_prompt(context['recent_transactions'])}

User Query: {message}

Provide a helpful, concise response focusing on their financial situation."""

        response = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=512,
            system="You are a helpful financial assistant focused on providing clear, actionable advice based on the user's financial data.",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text.strip()

    except Exception as e:
        logger.error(f"Error generating AI response: {str(e)}")
        return "I apologize, but I'm having trouble generating a response right now. Please try again in a moment."

def format_transactions_for_prompt(transactions: List[Dict]) -> str:
    """Format recent transactions for the AI prompt."""
    return "\n".join([
        f"- {tx['date']}: {tx['description']} (${tx['amount']:,.2f})"
        for tx in transactions
    ])