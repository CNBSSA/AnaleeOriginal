/**
 * explanationSuggestions.js — ESF (Explanation Suggestion Feature)
 */

import { apiFetch } from './api.js';

export class ExplanationSuggestionHandler {
    initialize() {
        document.querySelectorAll('.suggest-explanation-btn').forEach((button) => {
            button.addEventListener('click', () => this.handleClick(button));
        });
    }

    async handleClick(button) {
        const transactionId = button.dataset.transactionId;
        const description = button.dataset.description;
        const textarea = document.querySelector(`textarea[name="explanation_${transactionId}"]`);
        const suggestionDiv = document.getElementById(`explanation-suggestions-${transactionId}`);

        try {
            button.disabled = true;
            button.innerHTML = '<span class="spinner-border spinner-border-sm"></span>';

            const result = await apiFetch('/analyze/suggest-explanation', {
                method: 'POST',
                body: JSON.stringify({ description }),
            });

            if (!result?.success || !result.suggestion) {
                throw new Error(result?.error || 'No explanation suggestion available');
            }

            if (textarea) {
                textarea.value = result.suggestion;
                textarea.classList.add('has-content');
                textarea.dispatchEvent(new Event('input', { bubbles: true }));
            }

            if (suggestionDiv) {
                suggestionDiv.innerHTML = `
                    <div class="alert alert-success py-1 px-2 mb-0 small">
                        <i class="fas fa-lightbulb me-1"></i>${result.suggestion}
                    </div>`;
                suggestionDiv.style.display = 'block';
            }
        } catch (error) {
            console.error('ESF error:', error);
            if (suggestionDiv) {
                suggestionDiv.innerHTML = `
                    <div class="alert alert-danger py-1 px-2 mb-0 small">${error.message}</div>`;
                suggestionDiv.style.display = 'block';
            }
        } finally {
            button.disabled = false;
            button.innerHTML = '<i class="fas fa-lightbulb"></i> Suggest';
        }
    }
}
