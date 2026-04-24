"""
NLP utilities module — migrated from OpenAI to Anthropic Claude
"""
import os
import anthropic
import logging
from typing import Optional, Tuple, List
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

_claude_client: Optional[anthropic.Anthropic] = None

def get_claude_client() -> Optional[anthropic.Anthropic]:
    """Get cached Anthropic Claude client."""
    global _claude_client
    try:
        if _claude_client is not None:
            return _claude_client
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            logger.error("ANTHROPIC_API_KEY not found in environment")
            return None
        _claude_client = anthropic.Anthropic(api_key=api_key)
        logger.info("Anthropic Claude client initialized")
        return _claude_client
    except Exception as e:
        logger.error(f"Claude client initialization failed: {str(e)}")
        return None

# Backward-compatible alias — callers that import get_openai_client still work
get_openai_client = get_claude_client

CATEGORIES = [
    'income', 'groceries', 'utilities', 'transportation', 'entertainment',
    'shopping', 'healthcare', 'housing', 'education', 'investments',
    'dining', 'travel', 'insurance', 'personal_care', 'other'
]

def get_category_prompt(description: str) -> str:
    return f"""Categorize this financial transaction.

Transaction: {description}

Categories: {', '.join(CATEGORIES)}

Reply with ONLY this pipe-separated line (no other text):
category|confidence|explanation

Example: groceries|0.92|Supermarket food purchase"""


def categorize_transaction(description: str) -> Tuple[str, float, str]:
    """Categorize a single financial transaction using Claude."""
    if not description:
        return 'other', 0.1, "No description provided"

    client = get_claude_client()
    if not client:
        return 'other', 0.1, "AI service unavailable"

    try:
        response = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=150,
            system="You are a financial transaction categorization expert. Reply only with: category|confidence|explanation",
            messages=[{"role": "user", "content": get_category_prompt(description)}]
        )
        result = response.content[0].text.strip().split('|')
        if len(result) == 3:
            category = result[0].strip().lower()
            try:
                confidence = max(0.0, min(1.0, float(result[1].strip())))
            except ValueError:
                confidence = 0.5
            explanation = result[2].strip()
            if category not in CATEGORIES:
                category = 'other'
                confidence = 0.5
            return category, confidence, explanation
        return 'other', 0.1, "Unable to parse response"
    except Exception as e:
        logger.error(f"Error categorizing transaction: {str(e)}")
        return 'other', 0.1, f"Service error: {str(e)}"


def categorize_transactions_batch(descriptions: List[str]) -> List[Tuple[str, float, str]]:
    """Categorize multiple transactions in a single Claude call — far faster than one-by-one."""
    if not descriptions:
        return []

    client = get_claude_client()
    if not client:
        return [('other', 0.1, "AI service unavailable")] * len(descriptions)

    try:
        numbered = "\n".join(f"{i + 1}. {d}" for i, d in enumerate(descriptions))
        prompt = f"""Categorize each transaction below.

Categories: {', '.join(CATEGORIES)}

Transactions:
{numbered}

Reply with one line per transaction, exactly:
1|category|confidence|explanation
2|category|confidence|explanation
No other text."""

        response = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=max(512, len(descriptions) * 80),
            system="You are a financial transaction categorization expert. Reply only with numbered pipe-separated lines.",
            messages=[{"role": "user", "content": prompt}]
        )

        parsed: dict = {}
        for line in response.content[0].text.strip().split('\n'):
            parts = line.strip().split('|')
            if len(parts) == 4:
                try:
                    idx = int(parts[0].strip()) - 1
                    category = parts[1].strip().lower()
                    confidence = max(0.0, min(1.0, float(parts[2].strip())))
                    explanation = parts[3].strip()
                    if category not in CATEGORIES:
                        category = 'other'
                    parsed[idx] = (category, confidence, explanation)
                except (ValueError, IndexError):
                    continue

        return [parsed.get(i, ('other', 0.1, "Parse error")) for i in range(len(descriptions))]

    except Exception as e:
        logger.error(f"Batch categorization error: {str(e)}")
        return [('other', 0.1, f"Service error: {str(e)}")] * len(descriptions)
