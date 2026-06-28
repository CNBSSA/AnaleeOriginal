/**
 * explanationHandler.js
 *
 * Handles explanation textareas on the analyze page.
 */

export class ExplanationHandler {
    constructor() {
        this.initializeExplanationInputs();
    }

    initializeExplanationInputs() {
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
        container.appendChild(charCount);

        this.autoResize(textarea);
        this.updateCharCount(textarea.value, charCount);
        if (textarea.value.trim()) {
            textarea.classList.add('has-content');
        }

        textarea.addEventListener('input', () => {
            this.handleInput(textarea, charCount, () => timeoutId, (id) => { timeoutId = id; });
        });
        textarea.addEventListener('focus', () => {
            textarea.classList.add('explanation-focused');
            charCount.style.opacity = '1';
        });
        textarea.addEventListener('blur', () => {
            textarea.classList.remove('explanation-focused');
            charCount.style.opacity = '0.5';
            textarea.classList.toggle('has-content', textarea.value.trim() !== '');
        });
        textarea.addEventListener('keydown', (e) => this.handleKeydown(e));
    }

    autoResize(el) {
        el.style.height = 'auto';
        el.style.height = `${Math.min(200, Math.max(80, el.scrollHeight))}px`;
    }

    updateCharCount(text, charCount) {
        charCount.textContent = `${text.length}/500`;
    }

    handleInput(textarea, charCount, getTimeoutId, setTimeoutId) {
        this.autoResize(textarea);
        this.updateCharCount(textarea.value, charCount);
        textarea.classList.toggle('has-content', textarea.value.trim() !== '');

        clearTimeout(getTimeoutId());
        setTimeoutId(setTimeout(() => this.saveExplanation(textarea), 1000));
    }

    handleKeydown(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
        }
    }

    async saveExplanation(textarea) {
        try {
            const transactionId = textarea.dataset.transactionId;
            const description = textarea.dataset.description || textarea.closest('tr')?.querySelector('[data-description]')?.dataset.description;

            const response = await fetch('/update_explanation', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    transaction_id: transactionId,
                    explanation: textarea.value.trim(),
                    description,
                }),
            });

            if (!response.ok) {
                throw new Error('Failed to save explanation');
            }

            const result = await response.json();
            if (result.similar_transactions?.length > 0) {
                await this.handleSimilarTransactions(result.similar_transactions, textarea);
            }
        } catch (error) {
            console.error('Error saving explanation:', error);
        }
    }

    async handleSimilarTransactions(similarTransactions, sourceTextarea) {
        const shouldApplyToAll = confirm(
            `Found ${similarTransactions.length} similar transaction(s). Apply this explanation to them as well?`
        );

        if (!shouldApplyToAll) {
            return;
        }

        for (const similar of similarTransactions) {
            const similarTextarea = document.querySelector(`textarea[name="explanation_${similar.id}"]`);
            if (similarTextarea) {
                similarTextarea.value = sourceTextarea.value;
                similarTextarea.classList.add('has-content');
                await this.saveExplanation(similarTextarea);
            }
        }
    }
}

export default new ExplanationHandler();
