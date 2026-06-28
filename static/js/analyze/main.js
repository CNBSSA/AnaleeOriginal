/**
 * main.js
 *
 * Main entry point for the analyze page functionality.
 */

import ExplanationHandler from './explanationHandler.js';
import AccountSuggestionHandler from './accountSuggestions.js';
import TutorialManager from './tutorial.js';
import SimilarTransactionHandler from './similarTransactions.js';

class AnalyzeApplication {
    constructor() {
        this.initialized = false;
        this.form = null;
    }

    async initialize() {
        if (this.initialized) {
            return;
        }

        try {
            this.form = document.getElementById('analyzeForm');
            if (!this.form) {
                return;
            }

            this.explanationHandler = ExplanationHandler;
            this.accountSuggestionHandler = AccountSuggestionHandler;
            this.tutorialManager = TutorialManager;
            this.similarTransactionHandler = SimilarTransactionHandler;

            this.setupEventListeners();
            this.initializeTooltips();
            this.initialized = true;
        } catch (error) {
            console.error('Error initializing application:', error);
            this.showErrorMessage('Failed to initialize application. Please refresh the page.');
        }
    }

    setupEventListeners() {
        if (this.form) {
            this.form.addEventListener('submit', this.handleFormSubmission);
        }

        document.querySelectorAll('.account-select').forEach((select) => {
            select.addEventListener('change', (event) => this.saveAccountSelection(event.target));
        });

        window.addEventListener('error', this.handleGlobalError.bind(this));
    }

    async saveAccountSelection(select) {
        const transactionId = select.dataset.transactionId;
        if (!transactionId || !select.value) {
            return;
        }

        const textarea = document.querySelector(`textarea[name="explanation_${transactionId}"]`);

        try {
            const response = await fetch(`/analyze/save-transaction/${transactionId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    account_id: parseInt(select.value, 10),
                    explanation: textarea ? textarea.value.trim() : '',
                }),
            });

            if (!response.ok) {
                throw new Error('Failed to save account');
            }

            select.classList.add('border-success');
        } catch (error) {
            console.error('Error saving account:', error);
            this.showErrorMessage('Could not save account selection. Use Save Changes to retry.');
        }
    }

    initializeTooltips() {
        document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach((el) => {
            new bootstrap.Tooltip(el);
        });
    }

    handleFormSubmission() {
        // Allow normal POST save for the current page.
    }

    handleGlobalError(event) {
        console.error('Global error:', event.error);
    }

    showErrorMessage(message) {
        const toastElement = document.createElement('div');
        toastElement.className = 'toast position-fixed bottom-0 end-0 m-3';
        toastElement.innerHTML = `
            <div class="toast-header bg-danger text-white">
                <i class="fas fa-exclamation-circle me-2"></i>
                <strong class="me-auto">Error</strong>
                <button type="button" class="btn-close btn-close-white" data-bs-dismiss="toast"></button>
            </div>
            <div class="toast-body">${message}</div>`;
        document.body.appendChild(toastElement);
        const toast = new bootstrap.Toast(toastElement);
        toast.show();
        toastElement.addEventListener('hidden.bs.toast', () => toastElement.remove());
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const app = new AnalyzeApplication();
    app.initialize().catch((error) => console.error('Failed to initialize application:', error));
});

export default AnalyzeApplication;
