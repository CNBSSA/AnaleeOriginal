/**
 * similarTransactions.js — ERF (Explanation Recognition Feature)
 */

import { debounce } from './debounce.js';
import { apiFetch } from './api.js';

export class SimilarTransactionHandler {
    initialize() {
        document.querySelectorAll('.explanation-input').forEach((textarea) => {
            textarea.addEventListener(
                'input',
                debounce(() => this.handleSimilarTransactions(textarea), 500)
            );
        });
    }

    async handleSimilarTransactions(textarea) {
        const transactionId = textarea.dataset.transactionId;
        const description = textarea.dataset.description;
        const explanation = textarea.value.trim();
        const similarTransactionsDiv = document.getElementById(`similar-transactions-${transactionId}`);

        if (!explanation || !description || !similarTransactionsDiv) {
            if (similarTransactionsDiv) {
                this.hideSimilarTransactions(similarTransactionsDiv);
            }
            return;
        }

        try {
            const data = await apiFetch('/analyze/similar-transactions', {
                method: 'POST',
                body: JSON.stringify({ description, explanation }),
            });

            const similarTransactions = Array.isArray(data.similar_transactions)
                ? data.similar_transactions
                : [];

            this.displaySimilarTransactions(similarTransactions, similarTransactionsDiv, transactionId);
        } catch (error) {
            console.error('ERF error:', error);
            this.handleError(similarTransactionsDiv);
        }
    }

    displaySimilarTransactions(similarTransactions, container, sourceTransactionId) {
        if (!similarTransactions.length) {
            this.hideSimilarTransactions(container);
            return;
        }

        container.innerHTML = `
            <h6 class="text-muted">Similar transactions (ERF)</h6>
            <div class="list-group"></div>`;
        const listGroup = container.querySelector('.list-group');

        similarTransactions.forEach((transaction) => {
            const button = document.createElement('button');
            button.type = 'button';
            button.className = 'list-group-item list-group-item-action';
            button.innerHTML = `
                <div class="d-flex justify-content-between align-items-start">
                    <div>
                        <p class="mb-1">${transaction.description}</p>
                        <small class="text-muted">${transaction.explanation || ''}</small>
                    </div>
                    <span class="badge bg-primary">
                        ${Math.round((transaction.text_similarity || 0) * 100)}% match
                    </span>
                </div>`;
            button.addEventListener('click', () => {
                this.applyExplanationToCurrent(sourceTransactionId, transaction.explanation);
            });
            listGroup.appendChild(button);
        });

        container.style.display = 'block';
    }

    applyExplanationToCurrent(transactionId, explanation) {
        const textarea = document.querySelector(`textarea[name="explanation_${transactionId}"]`);
        if (!textarea) {
            return;
        }
        textarea.value = explanation;
        textarea.classList.add('has-content');
        textarea.dispatchEvent(new Event('input', { bubbles: true }));
    }

    hideSimilarTransactions(container) {
        container.style.display = 'none';
        container.innerHTML = '';
    }

    handleError(container) {
        container.innerHTML = `
            <div class="alert alert-warning py-1 px-2 mb-0 small">
                Could not find similar transactions right now.
            </div>`;
        container.style.display = 'block';
    }
}
