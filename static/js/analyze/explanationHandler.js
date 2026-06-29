/**
 * explanationHandler.js — saves explanations and triggers ERF lookups.
 */

import { apiFetch } from './api.js';

export class ExplanationHandler {
    initialize() {
        document.querySelectorAll('.explanation-input').forEach((textarea) => {
            this.setupTextarea(textarea);
        });
    }

    setupTextarea(textarea) {
        const container = textarea.closest('.explanation-container');
        if (!container) {
            return;
        }

        let timeoutId;
        const charCount = document.createElement('span');
        charCount.className = 'char-count position-absolute bottom-0 end-0 small text-muted pe-2';
        container.style.position = 'relative';
        container.appendChild(charCount);

        this.autoResize(textarea);
        this.updateCharCount(textarea.value, charCount);

        textarea.addEventListener('input', () => {
            this.autoResize(textarea);
            this.updateCharCount(textarea.value, charCount);
            textarea.classList.toggle('has-content', textarea.value.trim() !== '');
            clearTimeout(timeoutId);
            timeoutId = setTimeout(() => this.saveExplanation(textarea), 1000);
        });
    }

    autoResize(el) {
        el.style.height = 'auto';
        el.style.height = `${Math.min(200, Math.max(80, el.scrollHeight))}px`;
    }

    updateCharCount(text, charCount) {
        charCount.textContent = `${text.length}/500`;
    }

    async saveExplanation(textarea) {
        try {
            const transactionId = textarea.dataset.transactionId;
            const description = textarea.dataset.description;

            await apiFetch('/update_explanation', {
                method: 'POST',
                body: JSON.stringify({
                    transaction_id: transactionId,
                    explanation: textarea.value.trim(),
                    description,
                }),
            });
        } catch (error) {
            console.error('Error saving explanation:', error);
        }
    }
}
