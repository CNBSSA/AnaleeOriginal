"""The Practice Club — hub (member, seat) → local Analee user mapping.

One additive table; nothing on the existing models is modified. The table is
created idempotently by `club_sso.register()` (this repo deploys with
create_all(), not migrations — see app.py).
"""
from __future__ import annotations

from datetime import datetime

from models import db


class ClubMemberLink(db.Model):
    __tablename__ = "club_member_link"

    id = db.Column(db.Integer, primary_key=True)
    hub_member_id = db.Column(db.String(64), nullable=False)
    seat_id = db.Column(db.String(64), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship("User")

    __table_args__ = (
        db.UniqueConstraint("hub_member_id", "seat_id", name="uq_club_member_seat"),
    )

    def __repr__(self):
        return f"<ClubMemberLink member={self.hub_member_id} seat={self.seat_id}>"
