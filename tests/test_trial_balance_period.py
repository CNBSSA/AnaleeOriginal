"""
Regression test for the trial balance period-filtering bug.

The trial balance report (reports/routes.py::trial_balance) joins accounts to
their transactions filtered by the financial-year date range, then computes each
account's balance with ``sum(t.amount for t in account.transactions)``.

Because ``Account.transactions`` is a lazy relationship, the date filter in the
query's WHERE clause only decides *which accounts* are returned -- it does NOT
restrict what ``account.transactions`` yields. Without ``contains_eager`` the
relationship lazy-loads every transaction for the account regardless of date, so
the balance silently includes out-of-period transactions and the trial balance
totals are wrong.

This test reproduces the relationship/query shape in isolation and asserts that
the date-filtered query populated via ``contains_eager`` sums only in-period
transactions.
"""
from datetime import date

from sqlalchemy import create_engine, Column, Integer, String, Float, Date, ForeignKey
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, contains_eager

Base = declarative_base()


class _Account(Base):
    __tablename__ = "account"
    id = Column(Integer, primary_key=True)
    link = Column(String)
    name = Column(String)
    transactions = relationship("_Transaction", back_populates="account")


class _Transaction(Base):
    __tablename__ = "transaction"
    id = Column(Integer, primary_key=True)
    date = Column(Date)
    amount = Column(Float)
    account_id = Column(Integer, ForeignKey("account.id"))
    account = relationship("_Account", back_populates="transactions")


def _make_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()

    acc = _Account(link="1000", name="Office Supplies")
    session.add(acc)
    session.flush()
    # $100 inside FY2026, $500 outside (FY2025) -> period balance must be 100.
    session.add(_Transaction(date=date(2026, 3, 1), amount=100.0, account_id=acc.id))
    session.add(_Transaction(date=date(2025, 3, 1), amount=500.0, account_id=acc.id))
    session.commit()
    return session


_START, _END = date(2026, 1, 1), date(2026, 12, 31)


def _period_balance(accounts):
    total = 0.0
    for account in accounts:
        total += sum(t.amount for t in account.transactions)
    return total


def test_trial_balance_sums_only_in_period_transactions():
    """The fixed query (with contains_eager) sums only in-period transactions."""
    session = _make_session()
    accounts = (
        session.query(_Account)
        .outerjoin(_Account.transactions)
        .filter((_Transaction.date >= _START) & (_Transaction.date <= _END))
        .options(contains_eager(_Account.transactions))
        .order_by(_Account.link)
        .all()
    )
    assert _period_balance(accounts) == 100.0


def test_lazy_relationship_without_contains_eager_is_wrong():
    """Documents the bug: without contains_eager the out-of-period $500 leaks in."""
    session = _make_session()
    accounts = (
        session.query(_Account)
        .outerjoin(_Account.transactions)
        .filter((_Transaction.date >= _START) & (_Transaction.date <= _END))
        .order_by(_Account.link)
        .all()
    )
    # The lazy relationship ignores the date filter -> 100 + 500.
    assert _period_balance(accounts) == 600.0


if __name__ == "__main__":
    test_trial_balance_sums_only_in_period_transactions()
    test_lazy_relationship_without_contains_eager_is_wrong()
    print("All trial-balance period regression tests passed.")
