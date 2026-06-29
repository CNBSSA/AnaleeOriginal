"""Tests for entity type lock and chart refresh rules."""
from datetime import datetime

import pytest

from models import Account, CompanySettings, Entity, Transaction, db
from services.chart_of_accounts import seed_admin_charts, seed_entities, set_entity_for_user
from services.entity_chart_rules import (
    EntityChangeBlocked,
    apply_entity_change,
    provision_chart_if_missing,
    user_has_posted_transactions,
)


@pytest.fixture
def chart_user(app, sample_user):
    with app.app_context():
        seed_entities()
        seed_admin_charts()
    return sample_user


def test_provision_chart_if_missing_backfills_existing_user(app, chart_user):
    with app.app_context():
        private = Entity.query.filter_by(name='Private Company').first()
        settings = CompanySettings(
            user_id=chart_user,
            company_name='Backfill Co',
            financial_year_end=2,
            entity_id=private.id,
        )
        db.session.add(settings)
        db.session.commit()

        count = provision_chart_if_missing(chart_user)
        assert count > 0
        assert Account.query.filter_by(user_id=chart_user).count() == count


def test_entity_switch_rebuilds_chart_without_transactions(app, chart_user):
    with app.app_context():
        private = Entity.query.filter_by(name='Private Company').first()
        npo = Entity.query.filter_by(name='NPO').first()

        apply_entity_change(chart_user, private.id)
        private_count = Account.query.filter_by(user_id=chart_user).count()

        added, rebuilt = apply_entity_change(chart_user, npo.id)
        assert rebuilt is True
        npo_count = Account.query.filter_by(user_id=chart_user).count()
        assert npo_count > 0
        assert npo_count != private_count or private_count == 0


def test_entity_change_blocked_after_transaction(app, chart_user):
    with app.app_context():
        private = Entity.query.filter_by(name='Private Company').first()
        npo = Entity.query.filter_by(name='NPO').first()

        apply_entity_change(chart_user, private.id)
        account = Account.query.filter_by(user_id=chart_user).first()
        db.session.add(Transaction(
            date=datetime.utcnow(),
            description='Test txn',
            amount=100.0,
            user_id=chart_user,
            account_id=account.id,
        ))
        db.session.commit()

        assert user_has_posted_transactions(chart_user) is True

        with pytest.raises(EntityChangeBlocked):
            apply_entity_change(chart_user, npo.id)

        settings = CompanySettings.query.filter_by(user_id=chart_user).first()
        assert settings.entity_id == private.id


def test_same_entity_save_is_idempotent(app, chart_user):
    with app.app_context():
        private = Entity.query.filter_by(name='Private Company').first()
        apply_entity_change(chart_user, private.id)
        count_after_first = Account.query.filter_by(user_id=chart_user).count()

        added, rebuilt = apply_entity_change(chart_user, private.id)
        assert rebuilt is False
        assert added == 0
        assert Account.query.filter_by(user_id=chart_user).count() == count_after_first
