"""One reading of a date, across every import path.

The defect these lock (2026-09-20): five import paths parsed dates their own
way. The main bank-statement path used dayfirst=True while excel_reader handed
it every cell as a string, so an ISO date — or a real Excel date cell, which
stringifies to '2025-04-03 00:00:00' — was read as YYYY-DD-MM and stored as
4 March. The other four used the pandas default, which reads the South African
'03/04/2025' as 4 March instead of 3 April.

Neither flag is right on its own, so these tests assert the ONLY thing that
matters: every way of writing the same day must produce the same day.
"""
import datetime

import pandas as pd
import pytest

from utils.date_parsing import (
    is_year_first, parse_statement_date, parse_statement_date_or_raise,
)

THE_THIRD_OF_APRIL = datetime.date(2025, 4, 3)

# Every shape a South African bank statement puts in a Date cell.
SAME_DAY_WRITTEN_MANY_WAYS = [
    '03/04/2025',              # day-first, the SA convention
    '03-04-2025',
    '3/4/2025',
    '2025-04-03',              # ISO — dayfirst=True read this as 4 March
    '2025/04/03',
    '2025-04-03 00:00:00',     # an Excel date cell, stringified by dtype=str
    '3 Apr 2025',
    datetime.date(2025, 4, 3),
    datetime.datetime(2025, 4, 3),
    pd.Timestamp('2025-04-03'),
]


@pytest.mark.parametrize('written', SAME_DAY_WRITTEN_MANY_WAYS)
def test_every_way_of_writing_the_third_of_april_gives_the_third_of_april(written):
    parsed = parse_statement_date(written)
    assert parsed is not None, f'{written!r} could not be read at all'
    assert parsed.date() == THE_THIRD_OF_APRIL, (
        f'{written!r} was read as {parsed.date()}, not 3 April 2025 — '
        f'day and month are transposed')


def test_a_day_past_the_twelfth_is_unambiguous_either_way():
    """13/04 cannot be month-first, so it was always read correctly. It is the
    control: a fix that changed THIS would be breaking something that worked."""
    assert parse_statement_date('13/04/2025').date() == datetime.date(2025, 4, 13)
    assert parse_statement_date('2025-04-13').date() == datetime.date(2025, 4, 13)


def test_the_two_conventions_disagree_only_where_they_must():
    """A sanity check on the premise: 03/04 really is ambiguous, and the old
    default really did resolve it the wrong way for South Africa."""
    assert pd.to_datetime('03/04/2025').date() == datetime.date(2025, 3, 4)
    assert parse_statement_date('03/04/2025').date() == datetime.date(2025, 4, 3)


def test_iso_is_never_read_day_first():
    """The half of the defect that dayfirst=True CAUSED rather than fixed."""
    assert pd.to_datetime('2025-04-03', dayfirst=True).date() == datetime.date(2025, 3, 4)
    assert parse_statement_date('2025-04-03').date() == THE_THIRD_OF_APRIL


@pytest.mark.parametrize('text,expected', [
    ('2025-04-03', True), ('2025/4/3', True), ('2025.04.03', True),
    ('2025-04-03 09:15', True),
    ('03/04/2025', False), ('3 Apr 2025', False), ('', False),
])
def test_year_first_detection(text, expected):
    assert is_year_first(text) is expected


@pytest.mark.parametrize('junk', [None, '', '   ', 'not a date', float('nan')])
def test_unreadable_values_return_none_rather_than_raising(junk):
    assert parse_statement_date(junk) is None


def test_or_raise_names_the_value_and_the_accepted_forms():
    with pytest.raises(ValueError) as exc:
        parse_statement_date_or_raise('not a date')
    message = str(exc.value)
    assert 'not a date' in message
    assert '03/04/2025' in message and '2025-04-03' in message


def test_a_real_date_object_is_never_reparsed():
    """Re-parsing a date through its string form is how day and month get
    swapped, so a value that is already a date must pass through untouched."""
    for value in (datetime.date(2025, 4, 3), datetime.datetime(2025, 4, 3, 9, 15),
                  pd.Timestamp('2025-04-03 09:15')):
        assert parse_statement_date(value).date() == THE_THIRD_OF_APRIL


class TestTheImportPathsAgree:
    """The point of a shared parser: the paths must no longer disagree."""

    def test_the_bank_statement_detector_reads_both_conventions(self):
        from bank_statements.format_detector import (
            normalize_bank_statement_dataframe,
        )
        for written in ('03/04/2025', '2025-04-03', '2025-04-03 00:00:00'):
            raw = pd.DataFrame([
                ['Date', 'Description', 'Amount'],
                [written, 'Client deposit', '1500.00'],
            ])
            out = normalize_bank_statement_dataframe(raw)
            assert out.iloc[0]['Date'].date() == THE_THIRD_OF_APRIL, (
                f'the detector read {written!r} as '
                f'{out.iloc[0]["Date"].date()}')

    def test_a_statement_month_survives_the_round_trip(self):
        """The whole March statement, as an ISO file — the case that was being
        scattered across twelve months."""
        from bank_statements.format_detector import (
            normalize_bank_statement_dataframe,
        )
        days = [1, 2, 5, 9, 12, 13, 20, 28]
        rows = [['Date', 'Description', 'Amount']]
        rows += [[f'2025-03-{d:02d}', f'Txn {d}', '100.00'] for d in days]
        out = normalize_bank_statement_dataframe(pd.DataFrame(rows))
        got = sorted(d.date() for d in out['Date'])
        assert got == [datetime.date(2025, 3, d) for d in days], (
            'an ISO March statement did not come back as March')
        assert {d.month for d in out['Date']} == {3}, (
            'the statement was scattered across more than one month')
