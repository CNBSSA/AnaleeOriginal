"""ASF degradation guard — the boundary refuses a guessed account.

Festus, 2026-09-03, with a screenshot: he typed the explanation "Bank charges"
on a bank-charge line and Analee offered **"Patents and Trademarks -
Amortisation, 42% match"** — while a "Bank Charges" ledger account existed.

Reproduced exactly. When the AI call cannot be made or fails, the frozen engine
falls back to ``_basic_account_matching``: character similarity between
``"<description> - <explanation>"`` and ``"<account name> <category>"``. On a
real statement line the explanation is roughly a fifth of that string, so the
comparison is dominated by the bank's reference text:

    explanation alone      -> Bank Charges 0.73, Patents 0.25   (right answer)
    description + " - " + explanation -> Bank Charges 0.31, Patents 0.42  (the screenshot)

The matcher is unchanged since January 2025 — it is not a regression. What
changed is that it is now being REACHED, because the AI call is failing. So the
fix is not to the (frozen) matcher: it is to stop presenting its output as an
answer. These tests lock the three doors in ``routes.suggest_account``:

  1. no ANTHROPIC_API_KEY  -> decline, engine never called
  2. key set, client None  -> decline (the engine logs "AI client initialized
     successfully" either way, so the log cannot be trusted — the object is)
  3. engine returned its fallback -> decline

and that a genuine AI suggestion still passes through untouched.

The frozen engine (predictive_features.py, machine-locked) is NOT modified by
any of this — every guard lives at the route boundary.
"""
import os

import pytest

from routes import _asf_is_degraded, _ASF_FALLBACK_MARKER


EMAIL = "asfuser@example.com"
PASSWORD = "Sup3rSecret!"
URL = "/analyze/suggest-account"
PAYLOAD = {"description": "FEE IMMEDIATE PAYMENT 18H15 R90.00 #971",
           "explanation": "Bank charges"}


def _register_and_login(client):
    client.post("/auth/register", data={
        "username": "asfuser", "email": EMAIL,
        "password": PASSWORD, "confirm_password": PASSWORD,
    })
    client.post("/auth/login", data={"email": EMAIL, "password": PASSWORD})


# ---- the pure detector -----------------------------------------------------

def test_detector_recognises_the_engine_fallback():
    """The fallback's own reasoning string is the signal."""
    fallback = {
        'success': True,
        'account': 'Patents and Trademarks - Amortisation',
        'confidence': 0.42,
        'reasoning': (f'{_ASF_FALLBACK_MARKER} | Similarity score: 0.42 | '
                      'Matched against: Patents and Trademarks - Amortisation '
                      '(Non-Current Assets)'),
    }
    assert _asf_is_degraded(fallback) is True


def test_detector_passes_a_real_ai_answer():
    real = {'success': True, 'account': 'Bank Charges', 'confidence': 0.95,
            'reasoning': 'Bank fees charged by the institution are an expense.'}
    assert _asf_is_degraded(real) is False


def test_detector_handles_a_list_reasoning_and_junk():
    """Defensive: reasoning has been a list in this engine's history."""
    assert _asf_is_degraded({'reasoning': [_ASF_FALLBACK_MARKER, 'x']}) is True
    assert _asf_is_degraded({}) is False
    assert _asf_is_degraded({'reasoning': None}) is False
    assert _asf_is_degraded(None) is False
    assert _asf_is_degraded('not a dict') is False


# ---- door 1: no key --------------------------------------------------------

def test_no_api_key_declines_without_calling_the_engine(canary_app, monkeypatch):
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)

    called = []
    import predictive_features
    monkeypatch.setattr(predictive_features.PredictiveFeatures, 'suggest_account',
                        lambda self, *a, **k: called.append(1))

    client = canary_app.test_client()
    _register_and_login(client)
    body = client.post(URL, json=PAYLOAD).get_json()

    assert body['success'] is False
    assert body['ai_online'] is False
    assert 'offline' in body['message'].lower()
    assert called == [], "the frozen engine must not be called with no key"


# ---- door 2: key set, client did not construct -----------------------------

def test_key_set_but_client_none_declines(canary_app, monkeypatch):
    """The engine logs 'AI client initialized successfully' even when the
    helper returned None, so the guard checks the client, not the log."""
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-ant-looks-real-but-is-not')

    import predictive_features

    def _no_client(self):
        self.text_similarity_threshold = 0.70
        self.semantic_similarity_threshold = 0.95
        self.client = None

    monkeypatch.setattr(predictive_features.PredictiveFeatures, '__init__', _no_client)

    called = []
    monkeypatch.setattr(predictive_features.PredictiveFeatures, 'suggest_account',
                        lambda self, *a, **k: called.append(1))

    client = canary_app.test_client()
    _register_and_login(client)
    body = client.post(URL, json=PAYLOAD).get_json()

    assert body['success'] is False
    assert body['ai_online'] is False
    assert called == [], "engine must not be called when the client is None"


# ---- door 3: the engine answered, but from the fallback --------------------

def test_engine_fallback_result_is_declined_not_shown(canary_app, monkeypatch):
    """Festus's exact screenshot must never reach the user again."""
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-ant-present')

    import predictive_features

    def _has_client(self):
        self.client = object()

    monkeypatch.setattr(predictive_features.PredictiveFeatures, '__init__', _has_client)
    monkeypatch.setattr(
        predictive_features.PredictiveFeatures, 'suggest_account',
        lambda self, *a, **k: {
            'success': True,
            'account': 'Patents and Trademarks - Amortisation',
            'confidence': 0.42,
            'reasoning': (f'{_ASF_FALLBACK_MARKER} | Similarity score: 0.42 | '
                          'Matched against: Patents and Trademarks - '
                          'Amortisation (Non-Current Assets)'),
        })

    client = canary_app.test_client()
    _register_and_login(client)
    body = client.post(URL, json=PAYLOAD).get_json()

    assert body['success'] is False, "a text-match guess must not be served"
    assert body['ai_online'] is False
    assert 'Patents' not in str(body), "the wrong account must not reach the user"


def test_a_real_ai_suggestion_still_passes_through(canary_app, monkeypatch):
    """The guard must not break the working path."""
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-ant-present')

    import predictive_features
    monkeypatch.setattr(predictive_features.PredictiveFeatures, '__init__',
                        lambda self: setattr(self, 'client', object()))
    monkeypatch.setattr(
        predictive_features.PredictiveFeatures, 'suggest_account',
        lambda self, *a, **k: {
            'success': True, 'account': 'Bank Charges', 'confidence': 0.95,
            'reasoning': 'Bank fees are an operating expense.',
        })

    client = canary_app.test_client()
    _register_and_login(client)
    body = client.post(URL, json=PAYLOAD).get_json()

    assert body['success'] is True
    assert body['account'] == 'Bank Charges'
    assert body['confidence'] == 0.95


# ---- the coupling that let OCR work break analysis -------------------------

def test_claude_model_does_not_inherit_ocr_model(monkeypatch):
    """Tuning OCR_MODEL for statement reading must not silently re-point the
    analysis engine. That coupling is how a valid-looking deploy left ASF
    calling a model the key could not serve — every call raised, the engine
    fell through to text matching, and no log said so."""
    import importlib
    import config

    monkeypatch.setenv('OCR_MODEL', 'claude-some-vision-only-model')
    monkeypatch.delenv('CLAUDE_MODEL', raising=False)
    reloaded = importlib.reload(config)
    try:
        assert reloaded.OCR_MODEL == 'claude-some-vision-only-model'
        assert reloaded.CLAUDE_MODEL != reloaded.OCR_MODEL
        assert reloaded.CLAUDE_MODEL == 'claude-sonnet-4-6'
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_claude_model_is_still_overridable_on_its_own(monkeypatch):
    import importlib
    import config

    monkeypatch.setenv('CLAUDE_MODEL', 'claude-opus-5')
    reloaded = importlib.reload(config)
    try:
        assert reloaded.CLAUDE_MODEL == 'claude-opus-5'
    finally:
        monkeypatch.undo()
        importlib.reload(config)
