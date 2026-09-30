/**
 * main.js — initializes ASF, ERF, and ESF on the analyze page after DOM ready.
 */

import { apiFetch, showSaveNotice } from './api.js';
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
            const result = await apiFetch(`/analyze/save-transaction/${transactionId}`, {
                method: 'POST',
                body: JSON.stringify({
                    account_id: parseInt(select.value, 10),
                    explanation: textarea ? textarea.value.trim() : '',
                }),
            });
            select.classList.add('border-success');
            select.classList.remove('border-danger');
            showSaveNotice(select, result && result.message && result.message.startsWith('Account saved.')
                ? result.message : 'Saved.', true);
        } catch (error) {
            console.error('Error saving account:', error);
            select.classList.add('border-danger');
            showSaveNotice(select, `Not saved — ${error.message || 'please try again.'}`);
        }
    }

    initializeTooltips() {
        document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach((el) => {
            new bootstrap.Tooltip(el);
        });
    }
}

export default AnalyzeApplication;
