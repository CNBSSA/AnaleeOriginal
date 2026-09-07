"""Phase 5 — trial balance API transmission and signed share links."""
from datetime import datetime
from decimal import Decimal

import pytest
from itsdangerous import BadSignature, SignatureExpired

from models import Account, CompanySettings, Transaction, User, db
from reports.tb_share_tokens import (
    DEFAULT_MAX_AGE_SECONDS,
    create_share_token,
    verify_share_token,
)
from reports.trial_balance_service import (
    TrialBalanceContext,
    TrialBalanceRow,
    build_trial_balance_payload,
    load_trial_balance,
)


def _seed_balanced_tb(app, user_id: int):
    with app.app_context():
        settings = CompanySettings(
            user_id=user_id,
            company_name='ACME Pty Ltd',
            registration_number='2020/123456/07',
            financial_year_end=2,
        )
        bank = Account(
            link='ca.810.001',
            name='Bank Cheque Account 1',
            category='Assets',
            sub_category='Current Asset',
            user_id=user_id,
        )
        sales = Account(
            link='i.100.000',
            name='Sales',
            category='Income',
            sub_category='Income',
            user_id=user_id,
        )
        db.session.add_all([settings, bank, sales])
        db.session.flush()
        db.session.add_all([
            Transaction(
                date=datetime(2026, 4, 15),
                description='Receipt',
                amount=100.0,
                user_id=user_id,
                account_id=bank.id,
            ),
            Transaction(
                date=datetime(2026, 4, 16),
                description='Sale',
                amount=-100.0,
                user_id=user_id,
                account_id=sales.id,
            ),
        ])
        db.session.commit()


def test_build_trial_balance_payload_contract():
    ctx = TrialBalanceContext(
        accounts=(),
        start_date=datetime(2025, 3, 1),
        end_date=datetime(2026, 2, 28),
        total_debits=Decimal('100.00'),
        total_credits=Decimal('100.00'),
        rows=(
            TrialBalanceRow('ca.810.001', 'Bank Cheque Account 1', Decimal('100.00')),
            TrialBalanceRow('i.100.000', 'Sales', Decimal('-100.00')),
        ),
    )
    payload = build_trial_balance_payload(
        ctx,
        user_id=42,
        company_name='ACME Pty Ltd',
        registration_number='2020/123456/07',
    )
    assert payload['format_version'] == 1
    assert payload['source'] == 'analee'
    assert payload['company_id'] == 42
    assert payload['as_at'] == '2026-02-28'
    assert payload['balanced'] is True
    assert payload['rows'] == [
        {'link': 'ca.810.001', 'name': 'Bank Cheque Account 1', 'amount': 100.0},
        {'link': 'i.100.000', 'name': 'Sales', 'amount': -100.0},
    ]


def test_share_token_round_trip():
    token = create_share_token(99, secret_key='test-secret')
    assert verify_share_token(token, secret_key='test-secret') == 99


def test_share_token_rejects_tampering():
    token = create_share_token(1, secret_key='test-secret')
    with pytest.raises(BadSignature):
        verify_share_token(token + 'x', secret_key='test-secret')


def test_share_token_expires(monkeypatch):
    token = create_share_token(5, secret_key='test-secret')
    monkeypatch.setattr(
        'reports.tb_share_tokens._serializer',
        lambda secret_key: __import__('itsdangerous', fromlist=['URLSafeTimedSerializer']).URLSafeTimedSerializer(
            secret_key, salt='analee-tb-share'
        ),
    )
    with pytest.raises(SignatureExpired):
        verify_share_token(token, secret_key='test-secret', max_age=-1)


def test_load_trial_balance_payload_integration(app, sample_user):
    _seed_balanced_tb(app, sample_user)
    with app.app_context():
        ctx = load_trial_balance(sample_user)
        payload = build_trial_balance_payload(
            ctx,
            user_id=sample_user,
            company_name='ACME Pty Ltd',
            registration_number='2020/123456/07',
        )
        assert len(payload['rows']) == 2
        assert payload['balanced'] is True


@pytest.fixture
def transmission_client(app):
    """Flask test client with reports routes and login."""
    from flask_login import LoginManager

    from reports import reports

    login_manager = LoginManager()
    login_manager.init_app(app)
    app.register_blueprint(reports)

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    return app.test_client()


def test_api_trial_balance_requires_login(transmission_client):
    response = transmission_client.get('/api/trial-balance')
    assert response.status_code in (302, 401)


def test_api_trial_balance_json(transmission_client, app, sample_user):
    _seed_balanced_tb(app, sample_user)
    with app.app_context():
        user = User.query.get(sample_user)
        with transmission_client.session_transaction() as sess:
            sess['_user_id'] = str(user.id)
            sess['_fresh'] = True
        response = transmission_client.get('/api/trial-balance')
        assert response.status_code == 200
        data = response.get_json()
        assert data['company_id'] == sample_user
        assert data['rows'][0]['link'] == 'ca.810.001'


def test_shared_trial_balance_via_token(transmission_client, app, sample_user):
    _seed_balanced_tb(app, sample_user)
    token = create_share_token(sample_user, secret_key=app.config['SECRET_KEY'])
    response = transmission_client.get(f'/api/trial-balance/shared/{token}')
    assert response.status_code == 200
    data = response.get_json()
    assert data['company_name'] == 'ACME Pty Ltd'
    assert len(data['rows']) == 2


def _seed_prior_year_tb(app, user_id: int):
    """One balanced pair dated inside the PREVIOUS financial year (Mar 2025 –
    Feb 2026 for the Feb year-end that ``_seed_balanced_tb`` configures), with
    a different amount so the two years cannot be confused."""
    with app.app_context():
        bank = Account.query.filter_by(user_id=user_id, link='ca.810.001').first()
        sales = Account.query.filter_by(user_id=user_id, link='i.100.000').first()
        db.session.add_all([
            Transaction(date=datetime(2025, 6, 1), description='Old receipt',
                        amount=40.0, user_id=user_id, account_id=bank.id),
            Transaction(date=datetime(2025, 6, 2), description='Old sale',
                        amount=-40.0, user_id=user_id, account_id=sales.id),
        ])
        db.session.commit()


def test_shared_trial_balance_serves_the_closed_year_when_asked(transmission_client, app, sample_user):
    """QA #6: the share link could only ever serve the client's CURRENT year,
    so during AFS season THE ACCOUNTANTS received next year's year-to-date
    balance. ``as_at`` — the consumer's own year-end — now selects the year
    that closed; ``financial_year`` (start year) resolves to the same year."""
    _seed_balanced_tb(app, sample_user)
    _seed_prior_year_tb(app, sample_user)
    token = create_share_token(sample_user, secret_key=app.config['SECRET_KEY'])

    closed = transmission_client.get(f'/api/trial-balance/shared/{token}?as_at=2026-02-28')
    assert closed.status_code == 200
    data = closed.get_json()
    assert data['as_at'] == '2026-02-28'
    assert data['period_start'] == '2025-03-01'
    assert {r['link']: r['amount'] for r in data['rows']} == {'ca.810.001': 40.0, 'i.100.000': -40.0}

    by_year = transmission_client.get(f'/api/trial-balance/shared/{token}?financial_year=2025').get_json()
    assert by_year['as_at'] == '2026-02-28'
    assert len(by_year['rows']) == 2

    # Nothing asked → the current year, exactly as before.
    default = transmission_client.get(f'/api/trial-balance/shared/{token}').get_json()
    assert default['as_at'] != '2026-02-28'


def test_shared_trial_balance_rejects_a_malformed_as_at(transmission_client, app, sample_user):
    _seed_balanced_tb(app, sample_user)
    token = create_share_token(sample_user, secret_key=app.config['SECRET_KEY'])
    response = transmission_client.get(f'/api/trial-balance/shared/{token}?as_at=28/02/2026')
    assert response.status_code == 400
    assert 'YYYY-MM-DD' in response.get_json()['error']


def test_load_trial_balance_keywords_select_the_year_and_default_is_unchanged(app, sample_user):
    _seed_balanced_tb(app, sample_user)
    _seed_prior_year_tb(app, sample_user)
    with app.app_context():
        default = load_trial_balance(sample_user)
        chosen = load_trial_balance(sample_user, as_at=datetime(2026, 2, 28))
        assert chosen.end_date == datetime(2026, 2, 28)
        assert default.end_date != chosen.end_date
        assert {r.link: float(r.amount) for r in chosen.rows} == {'ca.810.001': 40.0, 'i.100.000': -40.0}
        assert load_trial_balance(sample_user, year=2025).end_date == chosen.end_date


def test_share_link_endpoint(transmission_client, app, sample_user):
    _seed_balanced_tb(app, sample_user)
    with app.app_context():
        with transmission_client.session_transaction() as sess:
            sess['_user_id'] = str(sample_user)
            sess['_fresh'] = True
        response = transmission_client.get('/api/trial-balance/share')
        assert response.status_code == 200
        data = response.get_json()
        assert 'share_url' in data
        assert data['expires_in_seconds'] == DEFAULT_MAX_AGE_SECONDS
