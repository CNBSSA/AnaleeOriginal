"""Regression tests for entity-scoped chart of accounts (BooksXperts parity)."""
from models import db, Entity, AdminChartOfAccounts, Account, CompanySettings, User
from services.chart_of_accounts import (
    seed_entities,
    seed_admin_charts,
    set_entity_for_user,
    provision_user_chart,
    DEFAULT_ENTITY_NAME,
)
from services.chart_seed_data import ENTITY_NAMES


def test_seed_entities_creates_five_sa_types(app):
    with app.app_context():
        entities = seed_entities()
        assert len(entities) == 5
        for name in ENTITY_NAMES:
            assert name in entities
        assert Entity.query.count() == 5


def test_seed_admin_charts_is_idempotent(app):
    with app.app_context():
        seed_entities()
        created_first, _ = seed_admin_charts()
        created_second, skipped_second = seed_admin_charts()
        assert created_first > 0
        assert created_second == 0
        assert skipped_second > 0


def test_same_link_allowed_across_different_entities(app):
    with app.app_context():
        seed_entities()
        entities = {e.name: e for e in Entity.query.all()}
        db.session.add(AdminChartOfAccounts(
            entity_id=entities['Private Company'].id,
            link='q.100.000',
            code='7000',
            name='Share Capital',
            category='Equity',
            sub_category='Equity',
        ))
        db.session.add(AdminChartOfAccounts(
            entity_id=entities['Close Corporation'].id,
            link='q.100.000',
            code='7000',
            name="Members' Contribution",
            category='Equity',
            sub_category='Equity',
        ))
        db.session.commit()
        assert AdminChartOfAccounts.query.filter_by(link='q.100.000').count() == 2


def test_set_entity_for_user_copies_master_chart(app, sample_user):
    with app.app_context():
        seed_entities()
        seed_admin_charts()
        private = Entity.query.filter_by(name='Private Company').first()
        added = set_entity_for_user(sample_user, private.id)
        assert added > 0
        user_accounts = Account.query.filter_by(user_id=sample_user).all()
        assert len(user_accounts) == added
        assert any(a.link == 'ca.810.001' for a in user_accounts)
        settings = CompanySettings.query.filter_by(user_id=sample_user).first()
        assert settings is not None
        assert settings.entity_id == private.id


def test_set_entity_for_user_is_idempotent(app, sample_user):
    with app.app_context():
        seed_entities()
        seed_admin_charts()
        private = Entity.query.filter_by(name='Private Company').first()
        first = set_entity_for_user(sample_user, private.id)
        second = set_entity_for_user(sample_user, private.id)
        assert first > 0
        assert second == 0
        assert Account.query.filter_by(user_id=sample_user).count() == first


def test_provision_user_chart_defaults_to_private_company(app, sample_user):
    with app.app_context():
        seed_entities()
        seed_admin_charts()
        added = provision_user_chart(sample_user)
        assert added > 0
        settings = CompanySettings.query.filter_by(user_id=sample_user).first()
        default_entity = Entity.query.filter_by(name=DEFAULT_ENTITY_NAME).first()
        assert settings.entity_id == default_entity.id


def test_entity_switch_adds_only_new_accounts(app, sample_user):
    with app.app_context():
        seed_entities()
        seed_admin_charts()
        private = Entity.query.filter_by(name='Private Company').first()
        npo = Entity.query.filter_by(name='NPO').first()
        private_added = set_entity_for_user(sample_user, private.id)
        npo_added = set_entity_for_user(sample_user, npo.id)
        assert private_added > 0
        assert npo_added > 0
        total = Account.query.filter_by(user_id=sample_user).count()
        assert total == private_added + npo_added
        npo_only = AdminChartOfAccounts.query.filter_by(
            entity_id=npo.id, link='i.600.000'
        ).first()
        assert npo_only is not None
        assert Account.query.filter_by(user_id=sample_user, link='i.600.000').first()


def test_user_create_default_accounts_provisions_chart(app):
    with app.app_context():
        seed_entities()
        seed_admin_charts()
        user = User(
            username='subscriber',
            email='sub@example.com',
            subscription_status='pending',
        )
        user.set_password('password')
        db.session.add(user)
        db.session.commit()
        added = User.create_default_accounts(user.id)
        assert added > 0
        assert Account.query.filter_by(user_id=user.id).count() == added
