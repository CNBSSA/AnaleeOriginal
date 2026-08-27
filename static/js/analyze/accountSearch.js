/**
 * Type-to-search on the account pickers (Festus, 2026-08-27: "I cannot search
 * for an account here"). Purely presentational: a small search box is added
 * above every .account-select / .analysis-select; typing filters the SAME
 * <option> list (case-insensitive contains). The select, its name, and the
 * posted value are unchanged — the frozen analysis engine is untouched.
 * Fail-soft: any error leaves the plain dropdowns exactly as they were.
 */
export default function enhanceAccountSelects(root = document) {
    try {
        root.querySelectorAll('select.account-select, select.analysis-select')
            .forEach((select) => {
                if (select.dataset.searchEnhanced) return;
                select.dataset.searchEnhanced = '1';

                const box = document.createElement('input');
                box.type = 'search';
                box.className = 'form-control form-control-sm mb-1 account-search-box';
                box.placeholder = 'Type to search accounts…';
                box.setAttribute('aria-label', 'Search accounts');
                select.parentNode.insertBefore(box, select);

                const options = Array.from(select.options);
                box.addEventListener('input', () => {
                    const q = box.value.trim().toLowerCase();
                    const selected = select.value;
                    // Rebuild the list from the ORIGINAL options each time so
                    // clearing the box restores everything (placeholder stays).
                    select.innerHTML = '';
                    options.forEach((opt) => {
                        const keep = !q
                            || opt.value === ''            // placeholder
                            || opt.value === selected      // never hide the choice
                            || opt.text.toLowerCase().includes(q);
                        if (keep) select.appendChild(opt);
                    });
                    select.value = selected;
                    // One real match left → preselect it so Enter/Tab commits.
                    const real = Array.from(select.options).filter(o => o.value !== '');
                    if (q && real.length === 1) {
                        select.value = real[0].value;
                        select.dispatchEvent(new Event('change', { bubbles: true }));
                    }
                });
            });
    } catch (e) {
        console.warn('account search enhancement skipped:', e);
    }
}
