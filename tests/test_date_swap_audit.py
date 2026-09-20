"""The transposition fingerprint — it must catch the real thing and, more
importantly, must not cry wolf over ordinary statements.

A false positive here costs an accountant an afternoon re-checking a statement
that was always fine, so the tests that matter most are the negative ones.
"""
import datetime

import pytest

from services.date_swap_audit import (
    MIN_DISTINCT_MONTHS, MIN_SHARE_ON_ONE_DAY, _assess,
)


def transpose(d):
    """What the old importer did to an ISO date: YYYY-MM-DD read as YYYY-DD-MM.
    Only possible while the day is 1-12 — beyond that the month is invalid and
    pandas fell back and read it correctly."""
    return datetime.date(d.year, d.day, d.month) if d.day <= 12 else d


class TestItCatchesTheRealThing:
    def test_a_transposed_march_statement_is_flagged(self):
        march = [datetime.date(2025, 3, d)
                 for d in (1, 2, 4, 6, 9, 11, 12, 15, 19, 23, 27, 30)]
        verdict = _assess([transpose(d) for d in march])
        assert verdict is not None
        assert verdict['distinct_months'] >= MIN_DISTINCT_MONTHS
        assert verdict['shared_day'] == 3, 'the shared day is the real month'
        assert verdict['likely_real_month'] == 3

    def test_it_names_the_real_month_so_the_file_can_be_found(self):
        july = [datetime.date(2025, 7, d) for d in (1, 3, 5, 8, 10, 12, 14, 21)]
        verdict = _assess([transpose(d) for d in july])
        assert verdict['likely_real_month'] == 7


class TestItDoesNotCryWolf:
    def test_an_ordinary_month_of_trading_is_not_flagged(self):
        april = [datetime.date(2025, 4, d)
                 for d in (1, 2, 3, 5, 8, 9, 12, 14, 15, 18, 22, 25, 28, 30)]
        assert _assess(april) is None

    def test_a_full_year_of_correct_trading_is_not_flagged(self):
        """Many months is not enough on its own — a year-long history spans
        twelve months and is perfectly healthy."""
        year = [datetime.date(2025, m, d)
                for m in range(1, 13) for d in (3, 11, 17, 24)]
        assert _assess(year) is None

    def test_a_monthly_debit_order_across_one_year_is_not_flagged(self):
        """The one real pattern that looks like the fingerprint: the same day
        every month. It must still clear, because the rows also have to be a
        MAJORITY of the file — and a statement is never only its debit order."""
        rent = [datetime.date(2025, m, 1) for m in range(1, 13)]
        other = [datetime.date(2025, m, d)
                 for m in range(1, 13) for d in (7, 14, 19, 26)]
        assert _assess(rent + other) is None

    def test_a_statement_inside_two_months_is_not_flagged(self):
        both = ([datetime.date(2025, 4, d) for d in (20, 24, 28, 30)]
                + [datetime.date(2025, 5, d) for d in (2, 5, 9, 14)])
        assert _assess(both) is None

    def test_too_few_rows_to_judge_is_not_flagged(self):
        assert _assess([datetime.date(2025, 4, 3)]) is None
        assert _assess([]) is None


class TestTheThresholdsAreTheOnesDocumented:
    def test_below_the_month_threshold_it_stays_quiet(self):
        dates = [datetime.date(2025, m, 4) for m in range(1, MIN_DISTINCT_MONTHS)]
        assert _assess(dates) is None

    def test_at_the_month_threshold_with_a_shared_day_it_speaks(self):
        dates = [datetime.date(2025, m, 4)
                 for m in range(1, MIN_DISTINCT_MONTHS + 1)]
        verdict = _assess(dates)
        assert verdict is not None
        assert verdict['share_on_shared_day'] == 100.0

    def test_the_shared_day_must_be_a_majority(self):
        """Spread across months but with no dominant day — that is a long
        trading history, not a transposition."""
        dates = [datetime.date(2025, m, d) for m in (1, 2, 3, 4) for d in (5, 6, 7, 8)]
        verdict = _assess(dates)
        assert verdict is None, (
            f'flagged a file whose top day holds only '
            f'{1/4:.0%} of rows, below the {MIN_SHARE_ON_ONE_DAY:.0%} bar')
