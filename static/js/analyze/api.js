/**
 * Shared fetch helper for analyze page API calls (includes CSRF token).
 */

export function getCsrfToken() {
    const fromMeta = document.querySelector('meta[name="csrf-token"]');
    if (fromMeta?.content) {
        return fromMeta.content;
    }

    const fromInput = document.querySelector('input[name="csrf_token"]');
    return fromInput?.value || '';
}

/**
 * Show a short save result right after a field (2026-09-30: failed saves were
 * logged to the console only, so an edit could be lost with nothing on screen).
 */
export function showSaveNotice(el, message, ok = false) {
    if (!el || !el.parentNode) return;
    let note = el.parentNode.querySelector(':scope > .save-notice');
    if (!note) {
        note = document.createElement('div');
        note.className = 'save-notice small mt-1';
        el.insertAdjacentElement('afterend', note);
    }
    note.className = `save-notice small mt-1 ${ok ? 'text-muted' : 'text-danger'}`;
    note.textContent = message;
}

export async function apiFetch(url, options = {}) {
    const headers = {
        'Content-Type': 'application/json',
        'X-CSRFToken': getCsrfToken(),
        ...(options.headers || {}),
    };

    const response = await fetch(url, { ...options, headers });
    let payload = null;

    try {
        payload = await response.json();
    } catch (error) {
        payload = null;
    }

    if (!response.ok) {
        const message = payload?.error || payload?.message || `Request failed (${response.status})`;
        throw new Error(message);
    }

    return payload;
}
