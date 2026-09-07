/**
 * "Apply to similar" — one accountant decision, applied to the whole family.
 *
 * Exceptions arrive in families (forty purchases at the same supermarket).
 * This previews the matching rows, shows exactly how many will change, and
 * only writes after the accountant confirms. Rows a person already decided are
 * never offered, and the server re-checks that on apply.
 */
import { apiFetch } from './api.js';

class FanoutHandler {
    bindUI() {
        document.querySelectorAll('.fanout-btn').forEach((button) => {
            button.addEventListener('click', () => this.run(button));
        });
    }

    note(transactionId, html, tone = 'text-muted') {
        const target = document.getElementById(`fanout-${transactionId}`);
        if (target) {
            target.innerHTML = `<span class="${tone}">${html}</span>`;
        }
    }

    async run(button) {
        const transactionId = button.dataset.transactionId;
        const fileId = button.dataset.fileId;

        const accountSelect = document.querySelector(
            `select[name="account_${transactionId}"]`);
        const explanationBox = document.querySelector(
            `textarea[name="explanation_${transactionId}"]`);
        const accountId = accountSelect ? accountSelect.value : '';
        const explanation = explanationBox ? explanationBox.value.trim() : '';

        if (!accountId && !explanation) {
            this.note(transactionId,
                'Set an account or write an explanation on this row first.',
                'text-warning');
            return;
        }

        button.disabled = true;
        this.note(transactionId, 'Looking for matching rows…');

        try {
            const preview = await apiFetch(
                `/analyze/${fileId}/similar-rows/${transactionId}`, { method: 'GET' });

            if (!preview.count) {
                this.note(transactionId, 'No other rows on this statement match this payee.');
                return;
            }

            const confirmed = window.confirm(
                `Apply this account and explanation to ${preview.count} other `
                + `row(s) matching "${preview.key}"?\n\n`
                + 'Rows already decided by you or the client are not included.');
            if (!confirmed) {
                this.note(transactionId, 'Nothing changed.');
                return;
            }

            const result = await apiFetch(`/analyze/${fileId}/apply-to-similar`, {
                method: 'POST',
                body: JSON.stringify({
                    transaction_ids: preview.rows.map((row) => row.id),
                    account_id: accountId || null,
                    explanation,
                }),
            });

            this.note(transactionId,
                `Applied to ${result.accounts_set} account(s) and `
                + `${result.explanations_set} explanation(s). Reload to see them.`,
                'text-success');
        } catch (error) {
            this.note(transactionId, `Could not apply: ${error.message}`, 'text-danger');
        } finally {
            button.disabled = false;
        }
    }
}

export default FanoutHandler;
