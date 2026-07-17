"""services.chart_reconciliation — additive-only sync from BooksXperts' live
seed (extends the detect-only guard in test_chart_reconciliation.py).

Fixture-based: writes a small synthetic BooksXperts seed file per test so the
diff classifier (ADD/CONFLICT/STALE/out-of-scope) and the apply-writer are
verified without depending on a real sibling checkout. A real-checkout smoke
test confirms the tool still runs cleanly against the actual repo (skipped
if no sibling checkout is present)."""
import ast
import os
import sys
import tempfile
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.chart_reconciliation import (  # noqa: E402
    apply_sync_diff,
    compute_sync_diff,
    default_booksxpert_root,
)

_FIXTURE_SEED = textwrap.dedent("""\
    CURRENT_ASSET = 1
    CURRENT_LIABILITY = 2
    COST_OF_SALES = 3
    EXPENSES = 4
    SALES = 5
    NON_CURRENT_ASSET = 6
    NON_CURRENT_LIABILITY = 7
    EQUITY = 8
    TAX = 9
    INCOME = 10

    COMMON_ACCOUNTS = [
        (9010, 'Zzfixture Bank Account', 'zzf.810.001', CURRENT_ASSET, '', 'df-cash'),
        (2000, 'Trade Payables', 'cl.500.000', CURRENT_LIABILITY, '', ''),
    ]

    ENTITY_SPECIFIC = {
        'Private Company': [
            (7000, 'Zzfixture Share Capital', 'zzf.999.000', EQUITY, '', ''),
        ],
        'Personal Liability Company': [
            (7000, 'Zzfixture Share Capital PLC', 'zzf.998.000', EQUITY, '', ''),
        ],
    }
    """)


def _write_fixture_seed(root, *, mutate=None):
    seed_dir = os.path.join(root, "app", "management", "commands")
    os.makedirs(seed_dir, exist_ok=True)
    text = _FIXTURE_SEED
    if mutate:
        text = mutate(text)
    with open(os.path.join(seed_dir, "seed_chart_of_accounts.py"), "w") as fh:
        fh.write(text)


def test_add_new_common_link():
    with tempfile.TemporaryDirectory() as tmp:
        _write_fixture_seed(tmp)
        diff = compute_sync_diff(tmp)
    links = {r[2] for r in diff.adds_common}
    assert "zzf.810.001" in links
    # cl.500.000 already exists in Analee's own COMMON_ACCOUNTS as 'Current Liability' -> no-op.
    assert "cl.500.000" not in links


def test_conflict_on_subcategory_mismatch():
    def mutate(text):
        return text.replace(
            "(2000, 'Trade Payables', 'cl.500.000', CURRENT_LIABILITY, '', ''),",
            "(2000, 'Trade Payables', 'cl.500.000', EXPENSES, '', ''),",
        )
    with tempfile.TemporaryDirectory() as tmp:
        _write_fixture_seed(tmp, mutate=mutate)
        diff = compute_sync_diff(tmp)
    links = {c[0] for c in diff.conflicts}
    assert "cl.500.000" in links
    assert "cl.500.000" not in {r[2] for r in diff.adds_common}


def test_stale_reported_not_removed():
    with tempfile.TemporaryDirectory() as tmp:
        _write_fixture_seed(tmp)
        diff = compute_sync_diff(tmp)
    # Analee's real chart carries plenty of links this tiny fixture doesn't —
    # e.g. na.010.000 — must be reported STALE, never removed from the file.
    assert "na.010.000" in diff.stale
    from services import chart_seed_data as seed
    assert any(t[2] == "na.010.000" for t in seed.COMMON_ACCOUNTS)


def test_out_of_scope_entity_is_never_an_add_target():
    with tempfile.TemporaryDirectory() as tmp:
        _write_fixture_seed(tmp)
        diff = compute_sync_diff(tmp)
    assert any(name == "Personal Liability Company" for name, _l, _n in diff.out_of_scope)
    assert "Personal Liability Company" not in diff.adds_by_entity


def test_allocated_account_numbers_are_unique_and_avoid_collisions():
    with tempfile.TemporaryDirectory() as tmp:
        _write_fixture_seed(tmp)
        diff = compute_sync_diff(tmp)
    from services import chart_seed_data as seed
    used = {t[0] for t in seed.COMMON_ACCOUNTS}
    for rows in seed.ENTITY_SPECIFIC.values():
        used.update(t[0] for t in rows)
    new_numbers = [r[0] for r in diff.adds_common]
    for rows in diff.adds_by_entity.values():
        new_numbers.extend(r[0] for r in rows)
    assert len(new_numbers) == len(set(new_numbers)), "duplicate numbers allocated within one diff"
    assert not (set(new_numbers) & used), "an allocated number collides with an existing one"


def test_apply_writes_only_add_rows_and_stays_valid_python():
    with tempfile.TemporaryDirectory() as tmp:
        _write_fixture_seed(tmp)
        diff = compute_sync_diff(tmp)

        target = os.path.join(tmp, "chart_seed_data_copy.py")
        from services import chart_seed_data as real_module
        with open(real_module.__file__) as fh:
            original = fh.read()
        with open(target, "w") as fh:
            fh.write(original)

        written = apply_sync_diff(diff, target)
        assert written == diff.add_count

        with open(target) as fh:
            new_source = fh.read()
        ast.parse(new_source)  # must still be valid Python
        assert "zzf.810.001" in new_source
        assert "zzf.999.000" in new_source  # Private Company entity add
        assert "zzf.998.000" not in new_source  # out-of-scope entity never written

        # And the module must still evaluate to the same shape it always has.
        namespace = {}
        exec(compile(new_source, target, "exec"), namespace)
        assert isinstance(namespace["COMMON_ACCOUNTS"], list)
        assert any(t[2] == "zzf.810.001" for t in namespace["COMMON_ACCOUNTS"])


def test_apply_is_noop_when_diff_has_no_adds():
    from services.chart_reconciliation import SyncDiff
    with tempfile.TemporaryDirectory() as tmp:
        target = os.path.join(tmp, "unused.py")
        with open(target, "w") as fh:
            fh.write("X = 1\n")
        written = apply_sync_diff(SyncDiff(), target)
    assert written == 0


def test_compute_diff_against_real_sibling_checkout_if_present():
    import pytest
    root = default_booksxpert_root()
    if not os.path.exists(os.path.join(root, "app", "management", "commands", "seed_chart_of_accounts.py")):
        pytest.skip("no sibling booksxpert checkout present")
    diff = compute_sync_diff(root)
    # After the 2026-07-17 sync, the real chart should have no unresolved
    # conflicts against the current BooksXperts seed.
    assert diff.conflicts == []
