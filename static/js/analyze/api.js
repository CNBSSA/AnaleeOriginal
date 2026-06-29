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
