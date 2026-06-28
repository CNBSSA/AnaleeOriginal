/**
 * Warn when categorising bank lines to accrual control or repairs expense accounts.
 */

const ACCRUAL_CONTROL_LINKS = new Set([
    'ca.300.000',
    'ca.300.001',
    'ca.330.000',
    'cl.500.000',
    'cl.600.000',
]);

const REPAIRS_EXPENSE_LINKS = new Set([
    'e.451.001',
    'e.451.002',
    'e.451.003',
    'e.451.004',
    'e.451.005',
]);

function isPpeCostLink(link) {
    return link.startsWith('na.') && !link.endsWith('.001') && !link.includes('Acc. Depr');
}

function showGuardrailAlert(select, message, alertClass) {
    const row = select.closest('tr');
    if (!row) {
        return;
    }
    const existing = row.querySelector('.cash-basis-guardrail');
    if (existing) {
        existing.remove();
    }
    if (!message) {
        return;
    }
    const cell = select.closest('td');
    const alert = document.createElement('div');
    alert.className = `alert ${alertClass} cash-basis-guardrail py-1 px-2 mt-1 mb-0 small`;
    alert.setAttribute('role', 'alert');
    alert.textContent = message;
    cell.appendChild(alert);
}

function guardrailMessage(link, label) {
    if (ACCRUAL_CONTROL_LINKS.has(link)) {
        return {
            text: `Cash-basis: "${label}" is an accrual control account. Use income or expense instead.`,
            alertClass: 'alert-warning',
        };
    }
    if (isPpeCostLink(link)) {
        return {
            text: `PPE cost account — BooksXperts or The Accountants will handle depreciation.`,
            alertClass: 'alert-info',
        };
    }
    if (REPAIRS_EXPENSE_LINKS.has(link)) {
        return {
            text: `Capital purchase? Use the asset **cost** account, not repairs expense. BooksXperts will depreciate.`,
            alertClass: 'alert-warning',
        };
    }
    return null;
}

export function bindCashBasisGuardrails() {
    document.querySelectorAll('.account-select').forEach((select) => {
        select.addEventListener('change', () => {
            const option = select.selectedOptions[0];
            if (!option || !option.dataset.link) {
                showGuardrailAlert(select, null);
                return;
            }
            const label = option.textContent.trim();
            const msg = guardrailMessage(option.dataset.link, label);
            showGuardrailAlert(select, msg ? msg.text : null, msg ? msg.alertClass : '');
        });
    });
}
