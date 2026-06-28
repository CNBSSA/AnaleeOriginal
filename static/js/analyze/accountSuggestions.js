/**
 * accountSuggestions.js
 *
 * AI-powered account suggestions for the analyze page.
 */

export class AccountSuggestionHandler {
    constructor() {
        this.initializeSuggestionButtons();
    }

    initializeSuggestionButtons() {
        document.querySelectorAll('.suggest-account-btn, .suggest-btn').forEach((button) => {
            button.addEventListener('click', async () => this.handleSuggestionClick(button));
        });
    }

    async handleSuggestionClick(button) {
        const transactionId = button.dataset.transactionId;
        const description = button.dataset.description;
        const explanation = button.dataset.explanation || '';
        const suggestionsDiv = document.getElementById(`suggestions-${transactionId}`);

        try {
            this.setLoadingState(button, suggestionsDiv);
            const suggestion = await this.fetchSuggestion(description, explanation);
            this.displaySuggestion(suggestion, suggestionsDiv, transactionId);
        } catch (error) {
            this.handleError(error, suggestionsDiv);
        } finally {
            this.resetButtonState(button);
        }
    }

    setLoadingState(button, suggestionsDiv) {
        button.disabled = true;
        button.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Loading...';
        if (suggestionsDiv) {
            suggestionsDiv.innerHTML = '<div class="alert alert-info mb-0">Loading suggestions...</div>';
            suggestionsDiv.style.display = 'block';
        }
    }

    resetButtonState(button) {
        button.disabled = false;
        button.innerHTML = '<i class="fas fa-magic"></i> AI Suggest';
    }

    async fetchSuggestion(description, explanation) {
        const response = await fetch('/analyze/suggest-account', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ description, explanation }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const suggestion = await response.json();
        if (suggestion.error) {
            throw new Error(suggestion.error);
        }
        return suggestion;
    }

    displaySuggestion(suggestion, suggestionsDiv, transactionId) {
        if (!suggestionsDiv) {
            return;
        }

        suggestionsDiv.innerHTML = '';
        suggestionsDiv.style.display = 'block';

        if (!suggestion || !suggestion.success) {
            suggestionsDiv.innerHTML = `
                <div class="alert alert-info mb-0">
                    <i class="fas fa-info-circle me-2"></i>
                    ${suggestion?.message || 'No suggestions available for this transaction'}
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
            option.text.toLowerCase().includes(suggestion.account.toLowerCase())
        );
        if (match) {
            select.value = match.value;
            select.classList.add('border-success');
        }
    }

    handleError(error, suggestionsDiv) {
        console.error('Error:', error);
        if (suggestionsDiv) {
            suggestionsDiv.innerHTML = `
                <div class="alert alert-danger mb-0">
                    <i class="fas fa-exclamation-circle me-2"></i>
                    Error getting suggestions: ${error.message}
                </div>`;
            suggestionsDiv.style.display = 'block';
        }
    }
}

export default new AccountSuggestionHandler();
