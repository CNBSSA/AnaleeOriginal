/**
 * accountSuggestions.js — ASF (Account Suggestion Feature)
 */

import { apiFetch } from './api.js';

export class AccountSuggestionHandler {
    initialize() {
        document.querySelectorAll('.suggest-account-btn, .suggest-btn').forEach((button) => {
            button.addEventListener('click', () => this.handleSuggestionClick(button));
        });
    }

    async handleSuggestionClick(button) {
        const transactionId = button.dataset.transactionId;
        const description = button.dataset.description;
        const explanationField = document.querySelector(`textarea[name="explanation_${transactionId}"]`);
        const explanation = explanationField?.value.trim() || button.dataset.explanation || '';
        const suggestionsDiv = document.getElementById(`suggestions-${transactionId}`);

        try {
            this.setLoadingState(button, suggestionsDiv);
            const suggestion = await apiFetch('/analyze/suggest-account', {
                method: 'POST',
                body: JSON.stringify({ description, explanation }),
            });
            this.displaySuggestion(suggestion, suggestionsDiv, transactionId);
        } catch (error) {
            this.handleError(error, suggestionsDiv);
        } finally {
            this.resetButtonState(button);
        }
    }

    setLoadingState(button, suggestionsDiv) {
        button.disabled = true;
        button.innerHTML = '<span class="spinner-border spinner-border-sm" role="status"></span> Loading...';
        if (suggestionsDiv) {
            suggestionsDiv.innerHTML = '<div class="alert alert-info mb-0">Loading suggestions...</div>';
            suggestionsDiv.style.display = 'block';
        }
    }

    resetButtonState(button) {
        button.disabled = false;
        button.innerHTML = '<i class="fas fa-magic"></i> ASF';
    }

    displaySuggestion(suggestion, suggestionsDiv, transactionId) {
        if (!suggestionsDiv) {
            return;
        }

        suggestionsDiv.innerHTML = '';
        suggestionsDiv.style.display = 'block';

        if (!suggestion?.success) {
            suggestionsDiv.innerHTML = `
                <div class="alert alert-info mb-0">
                    ${suggestion?.message || 'No account suggestions available'}
                </div>`;
            return;
        }

        const confidence = Math.round((suggestion.confidence || 0) * 100);
        const confidenceClass = confidence >= 80 ? 'success' : confidence >= 60 ? 'info' : 'warning';

        const item = document.createElement('button');
        item.type = 'button';
        item.className = 'list-group-item list-group-item-action suggestion-item';
        item.innerHTML = `
            <div class="d-flex justify-content-between align-items-start">
                <div>
                    <h6 class="mb-1">${suggestion.account}</h6>
                    <div class="suggestion-reasoning small">${suggestion.reasoning || ''}</div>
                </div>
                <span class="badge bg-${confidenceClass}">${confidence}% match</span>
            </div>`;
        item.addEventListener('click', () => this.applySuggestion(suggestion, transactionId));
        suggestionsDiv.appendChild(item);
    }

    applySuggestion(suggestion, transactionId) {
        const select = document.querySelector(`select[name="account_${transactionId}"]`);
        if (!select) {
            return;
        }

        const match = Array.from(select.options).find((option) =>
            option.text.toLowerCase().includes(String(suggestion.account).toLowerCase())
        );
        if (match) {
            select.value = match.value;
            select.dispatchEvent(new Event('change', { bubbles: true }));
            select.classList.add('border-success');
        }
    }

    handleError(error, suggestionsDiv) {
        console.error('ASF error:', error);
        if (suggestionsDiv) {
            suggestionsDiv.innerHTML = `
                <div class="alert alert-danger mb-0">${error.message}</div>`;
            suggestionsDiv.style.display = 'block';
        }
    }
}
