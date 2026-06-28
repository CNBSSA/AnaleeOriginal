/**
 * main.js — initializes ASF, ERF, and ESF on the analyze page after DOM ready.
 */

import { apiFetch } from './api.js';
import { ExplanationHandler } from './explanationHandler.js';
import { AccountSuggestionHandler } from './accountSuggestions.js';
import { ExplanationSuggestionHandler } from './explanationSuggestions.js';
import { SimilarTransactionHandler } from './similarTransactions.js';
import TutorialManager from './tutorial.js';
import { bindCashBasisGuardrails } from './cashBasisGuardrails.js';

class AnalyzeApplication {
    constructor() {
        this.initialized = false;
        this.form = null;
    }

    async initialize() {
        if (this.initialized) {
            return;
        }

        this.form = document.getElementById('analyzeForm');
        if (!this.form) {
            return;
        }

        this.explanationHandler = new ExplanationHandler();
        this.accountSuggestionHandler = new AccountSuggestionHandler();
        this.explanationSuggestionHandler = new ExplanationSuggestionHandler();
        this.similarTransactionHandler = new SimilarTransactionHandler();

        this.explanationHandler.initialize();
        this.accountSuggestionHandler.initialize();
        this.explanationSuggestionHandler.initialize();
        this.similarTransactionHandler.initialize();

        this.setupEventListeners();
        bindCashBasisGuardrails();
        this.initializeTooltips();
        this.initialized = true;
    }

    setupEventListeners() {
        document.querySelectorAll('.account-select').forEach((select) => {
            select.addEventListener('change', (event) => this.saveAccountSelection(event.target));
        });
    }

    async saveAccountSelection(select) {
        const transactionId = select.dataset.transactionId;
        if (!transactionId || !select.value) {
            return;
        }

        const textarea = document.querySelector(`textarea[name="explanation_${transactionId}"]`);

        try {
            await apiFetch(`/analyze/save-transaction/${transactionId}`, {
                method: 'POST',
                body: JSON.stringify({
                    account_id: parseInt(select.value, 10),
                    explanation: textarea ? textarea.value.trim() : '',
                }),
            });
            select.classList.add('border-success');
        } catch (error) {
            console.error('Error saving account:', error);
        }
    }

    initializeTooltips() {
        document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach((el) => {
            new bootstrap.Tooltip(el);
        });
    }
}

export default AnalyzeApplication;
