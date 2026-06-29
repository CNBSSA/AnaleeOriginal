"""No-login client ERF wizard (Flask blueprint)."""
from __future__ import annotations

import json
import logging

from flask import Blueprint, abort, current_app, jsonify, render_template, request, url_for
from itsdangerous import BadSignature, SignatureExpired

from client_explain_tokens import verify_client_explain_token
from models import Transaction, UploadedFile, db
from services import client_erf
from services.client_explanation import (
    SOURCE_CLIENT,
    SOURCE_CLIENT_ERF,
    get_file_for_owner,
    queue_counts,
    save_explanation,
    serialize_transaction,
    unexplained_client_queue,
)

logger = logging.getLogger(__name__)

client_explain_bp = Blueprint('client_explain', __name__)


def _resolve_file(token: str) -> tuple[UploadedFile, int]:
    try:
        file_id, user_id = verify_client_explain_token(
            token, secret_key=current_app.config['SECRET_KEY'],
        )
    except SignatureExpired as exc:
        abort(404, description='This link has expired. Ask your accountant for a new one.')
    except BadSignature as exc:
        abort(404, description='This link is not valid.')
    uploaded = get_file_for_owner(file_id, user_id)
    if uploaded is None:
        abort(404)
    return uploaded, user_id


def _wizard_context(uploaded, user_id, current, token, *, similar=None, saved_message=''):
    remaining, total = queue_counts(uploaded.id, user_id)
    explained = total - remaining
    return {
        'company_label': uploaded.filename or 'Bank statement',
        'transaction': serialize_transaction(current),
        'transaction_id': current.id,
        'token': token,
        'remaining': remaining,
        'total': total,
        'position': explained + 1,
        'saved_message': saved_message,
        'similar_rows': client_erf.serialize_similar(
            similar or [],
            reference_description=current.description or '',
        ),
    }


@client_explain_bp.route('/client-explain/<token>/', methods=['GET', 'POST'])
def client_explain(token: str):
    uploaded, user_id = _resolve_file(token)

    if request.method == 'POST' and request.form.get('action') == 'batch_apply':
        return _batch_apply(uploaded, user_id, token)

    if request.method == 'POST':
        but_id = request.form.get('transaction_id', type=int)
        explanation = (request.form.get('explanation') or '').strip()
        if not but_id or not explanation:
            abort(400, description='Explanation required.')
        current = Transaction.query.filter_by(
            id=but_id, file_id=uploaded.id, user_id=user_id,
        ).first()
        if current is None:
            abort(404)
        ok, reason = save_explanation(current, explanation, SOURCE_CLIENT)
        if not ok:
            abort(400, description=reason)
        db.session.commit()

        similar = client_erf.find_similar_unexplained(
            uploaded.id, user_id, current.description or '', exclude_id=current.id,
        )
        next_txn = unexplained_client_queue(uploaded.id, user_id).first()
        if next_txn is None:
            _, total = queue_counts(uploaded.id, user_id)
            return render_template('client_explain/done.html', company_label=uploaded.filename, total=total)
        return render_template(
            'client_explain/wizard.html',
            **_wizard_context(uploaded, user_id, next_txn, token, saved_message='Saved. Next transaction:'),
        )

    remaining, total = queue_counts(uploaded.id, user_id)
    if remaining == 0:
        return render_template('client_explain/done.html', company_label=uploaded.filename, total=total)
    current = unexplained_client_queue(uploaded.id, user_id).first()
    return render_template(
        'client_explain/wizard.html',
        **_wizard_context(uploaded, user_id, current, token),
    )


def _batch_apply(uploaded, user_id, token):
    if request.is_json:
        body = request.get_json(silent=True) or {}
    else:
        body = request.form
    source_id = body.get('source_transaction_id') or body.get('transaction_id')
    explanation = (body.get('explanation') or '').strip()
    target_ids = body.get('similar_ids') or body.get('target_ids') or []
    if isinstance(target_ids, str):
        try:
            target_ids = json.loads(target_ids)
        except json.JSONDecodeError:
            target_ids = [target_ids]
    if not source_id or not explanation:
        return jsonify({'ok': False, 'error': 'Missing data.'}), 400

    source = Transaction.query.filter_by(
        id=int(source_id), file_id=uploaded.id, user_id=user_id,
    ).first()
    if source is None:
        abort(404)
    allowed = {
        row.id for row in client_erf.find_similar_unexplained(
            uploaded.id, user_id, source.description or '', exclude_id=source.id,
        )
    }
    allowed.add(int(source_id))
    applied = 0
    for raw_id in target_ids:
        try:
            tid = int(raw_id)
        except (TypeError, ValueError):
            continue
        if tid not in allowed:
            continue
        txn = Transaction.query.filter_by(id=tid, file_id=uploaded.id, user_id=user_id).first()
        if txn is None:
            continue
        src = SOURCE_CLIENT if tid == int(source_id) else SOURCE_CLIENT_ERF
        ok, _ = save_explanation(txn, explanation, src)
        if ok:
            applied += 1
    db.session.commit()

    if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.is_json:
        next_txn = unexplained_client_queue(uploaded.id, user_id).first()
        return jsonify({
            'ok': True,
            'applied': applied,
            'done': next_txn is None,
            'next_url': url_for('client_explain.client_explain', token=token),
        })

    next_txn = unexplained_client_queue(uploaded.id, user_id).first()
    if next_txn is None:
        _, total = queue_counts(uploaded.id, user_id)
        return render_template('client_explain/done.html', company_label=uploaded.filename, total=total)
    return render_template(
        'client_explain/wizard.html',
        **_wizard_context(
            uploaded, user_id, next_txn, token,
            saved_message=f'Applied to {applied} similar transaction(s).',
        ),
    )


@client_explain_bp.route('/client-explain/<token>/similar/')
def client_explain_similar(token: str):
    uploaded, user_id = _resolve_file(token)
    description = (request.args.get('description') or '').strip()
    exclude_id = request.args.get('exclude_id', type=int)
    rows = client_erf.find_similar_unexplained(
        uploaded.id, user_id, description, exclude_id=exclude_id,
    )
    return jsonify({
        'similar': client_erf.serialize_similar(rows, reference_description=description),
        'count': len(rows),
    })


def build_client_explain_url(file_id: int, user_id: int) -> str:
    from client_explain_tokens import create_client_explain_token
    token = create_client_explain_token(
        file_id, user_id, secret_key=current_app.config['SECRET_KEY'],
    )
    return url_for('client_explain.client_explain', token=token, _external=True)
