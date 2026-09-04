"""A statement date we cannot read is REFUSED, never invented.

QA audit, 2026-09-04. ``ocr.confirm_receipt`` used to do this:

    try:
        date_value = datetime.strptime(raw_date, '%Y-%m-%d')
    except (TypeError, ValueError):
        date_value = datetime.utcnow()     # <-- silently "today"

so a row whose date failed to parse was imported stamped with whatever day the
import happened to run. The books looked complete, the transaction sat in the
wrong VAT/tax period, and nothing anywhere recorded that the date was made up.
Note the inconsistency it sat next to: an unreadable AMOUNT already dropped the
row. Only the date was invented.

Now the row is refused and named, so the user can correct it on the review
screen they just came from.
"""
import pytest

pytest.importorskip("flask_sqlalchemy")
pytest.importorskip("flask_login")

from flask import Flask
from flask_login import LoginManager
from models import db, User, Transaction
from ocr import ocr as ocr_bp


def _make_app():
    app = Flask(__name__, template_folder="templates")
    app.config.update(
        SQLALCHEMY_DATABASE_URI="sqlite:///:memory:",
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        SECRET_KEY="test",
        WTF_CSRF_ENABLED=False,
        LOGIN_DISABLED=True,
    )
    db.init_app(app)
    lm = LoginManager()
    lm.init_app(app)

    @lm.user_loader
    def _load(uid):
        return db.session.get(User, int(uid))

    app.register_blueprint(ocr_bp)
    app.add_url_rule('/upload', endpoint='main.upload', view_func=lambda: 'ok')
    return app


def _login(client, user_id):
    with client.session_transaction() as sess:
        sess['_user_id'] = str(user_id)


def _seed_user(app):
    with app.app_context():
        db.create_all()
        user = User(username='t', email='t@e.com', password_hash='x')
        db.session.add(user)
        db.session.commit()
        return user.id


def _confirm(client, dates, descriptions, amounts):
    return client.post('/ocr/statement/confirm', data={
        'date': dates, 'description': descriptions, 'amount': amounts,
        'filename': 'statement.pdf',
    }, follow_redirects=False)


def test_an_unreadable_date_is_never_stamped_with_today():
    app = _make_app()
    uid = _seed_user(app)
    client = app.test_client()
    _login(client, uid)

    _confirm(client,
             dates=['2026-03-15', 'not-a-date', '2026-03-17'],
             descriptions=['Good one', 'Bad date', 'Another good one'],
             amounts=['100.00', '250.00', '75.00'])

    with app.app_context():
        rows = Transaction.query.all()
        assert len(rows) == 2, "the unreadable row must not be imported"
        descriptions = {t.description for t in rows}
        assert descriptions == {'Good one', 'Another good one'}
        assert 'Bad date' not in descriptions, (
            "a row whose date could not be read was imported anyway — "
            "check it is not being given today's date")
        assert {t.date.strftime('%Y-%m-%d') for t in rows} == {
            '2026-03-15', '2026-03-17'}


def test_the_user_is_told_which_rows_were_refused():
    """Silently dropping the row would be its own defect — name it."""
    app = _make_app()
    uid = _seed_user(app)
    client = app.test_client()
    _login(client, uid)

    with client:
        _confirm(client,
                 dates=['2026-03-15', '15/03/2026', 'unknown'],
                 descriptions=['Good', 'Bad A', 'Bad B'],
                 amounts=['100.00', '250.00', '75.00'])
        from flask import get_flashed_messages
        messages = ' '.join(get_flashed_messages())

    assert 'Imported 1 transaction(s)' in messages
    assert 'NOT imported' in messages
    assert '2, 3' in messages, f"the refused rows must be named: {messages!r}"
    assert 'YYYY-MM-DD' in messages, "tell them the format that works"


def test_every_date_unreadable_imports_nothing_and_says_so():
    app = _make_app()
    uid = _seed_user(app)
    client = app.test_client()
    _login(client, uid)

    with client:
        _confirm(client,
                 dates=['nope', ''],
                 descriptions=['One', 'Two'],
                 amounts=['10.00', '20.00'])
        from flask import get_flashed_messages
        messages = ' '.join(get_flashed_messages())

    with app.app_context():
        assert Transaction.query.count() == 0

    assert 'Nothing was imported' in messages
    assert '1, 2' in messages


def test_a_clean_import_is_unchanged():
    """The working path must behave exactly as before."""
    app = _make_app()
    uid = _seed_user(app)
    client = app.test_client()
    _login(client, uid)

    with client:
        _confirm(client,
                 dates=['2026-03-15', '2026-03-16'],
                 descriptions=['One', 'Two'],
                 amounts=['10.00', '20.00'])
        from flask import get_flashed_messages
        messages = ' '.join(get_flashed_messages())

    with app.app_context():
        assert Transaction.query.count() == 2
    assert messages.strip() == 'Imported 2 transaction(s).'
    assert 'NOT imported' not in messages
