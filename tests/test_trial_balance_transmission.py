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
